use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use bevy_egui::{egui, EguiContextPass, EguiContexts, EguiPlugin};
use bevy_vector_shapes::prelude::*;
use rand::distr::weighted::WeightedIndex;
use rand::distr::Distribution;
use rand::{rng, Rng};

fn main() {
    // TODO: look into events wtf is this
    App::new()
        .add_plugins((
            DefaultPlugins,
            Shape2dPlugin::default()
        ))
        .add_plugins(EguiPlugin { enable_multipass_for_primary_context: true })
        .add_plugins((
            FrameTimeDiagnosticsPlugin::default(),
            LogDiagnosticsPlugin::default())
        )
        .insert_resource(AlgStats::default())
        .insert_resource(Config::default())
        .insert_resource(ClearColor(Color::srgb(196.0 / 255.0, 164.0 / 255.0, 132.0 / 255.0,)))
        .add_systems(Startup, (setup_camera, (setup_graph, setup_ants).chain()))
        .add_systems(Update, (
            draw_graph,
            (
                ant_behavior_system,
                pheromone_system
                    .run_if(run_if_ants_finished_round),
                kill_ants
                    .run_if(run_if_ants_finished_round),
                setup_ants
                    .run_if(run_if_ants_finished_round),
                movement_system
            ).chain().run_if(run_if_ants_finished_round.map(|b| !b))
        ))
        .add_systems(Update, (
            kill_ants,
            setup_graph,
            setup_ants)
                .chain()
                .run_if(run_if_graph_cfg_changed)
        )
        .add_systems(Update, draw_ants.run_if(run_if_should_visualize_ants))
        .add_systems(Update, (
            stats_system.run_if(run_if_ants_finished_round),
            reset_stats.run_if(run_if_graph_cfg_changed)
        ))
        .add_systems(EguiContextPass, ui_system)
        .run();
}

/*
--- RESOURCES ---
*/
#[derive(Resource)]
struct Config {
    node_count: usize,
    ant_count: i32,
    distance_power: f32,
    pheromone_power: f32,
    pheromone_intensity: f32,
    evaporation_rate: f32,
    should_visualize_ants: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            node_count: 6,
            ant_count: 50,
            distance_power: 3.0,
            pheromone_power: 6.0,
            pheromone_intensity: 1000.0,
            evaporation_rate: 0.3,
            should_visualize_ants: true,
        }
    }
}

#[derive(Resource)]
pub struct AlgStats {
    current_shortest_path_dist: f32,
    shortest_path_dist: f32,
}

impl Default for AlgStats {
    fn default() -> Self {
        Self {
            current_shortest_path_dist: f32::INFINITY,
            shortest_path_dist: f32::INFINITY,
        }
    }
}

/*
--- COMPONENTS ---
*/

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
struct NodeId(usize);

struct EdgeData {
    a: NodeId,
    b: NodeId,
    distance: f32,
    pheromone: f32,
}

#[derive(Resource)]
struct Graph {
    nodes: Vec<Vec2>,
    edges: Vec<EdgeData>,
}

impl Graph {
    fn pheromone_between(&self, x: NodeId, y: NodeId) -> f32 {
        self.edges.iter()
            .find(|e| (e.a == x && e.b == y) || (e.a == y && e.b == x))
            .map(|e| e.pheromone)
            .unwrap_or(0.0)
    }

    fn deposit_pheromone(&mut self, x: NodeId, y: NodeId, amount: f32) {
        if let Some(e) = self.edges.iter_mut()
            .find(|e| (e.a == x && e.b == y) || (e.a == y && e.b == x))
        {
            e.pheromone += amount;
        }
    }

    fn distance_between(&self, x: NodeId, y: NodeId) -> f32 {
        self.edges.iter()
            .find(|e| (e.a == x && e.b == y) || (e.a == y && e.b == x))
            .map(|e| e.distance)
            .unwrap_or(f32::INFINITY)
    }
}

#[derive(Component)]
struct Ant {
    current: NodeId,
    target: Option<NodeId>,
    visited: Vec<NodeId>,
    path: Vec<(NodeId, NodeId)>,
}

#[derive(Component)]
struct Movement {
    direction: Vec2,
    speed: f32
}

/*
--- SYSTEMS ---
*/

fn setup_camera(mut commands: Commands) {
    commands.spawn((
        Camera2d,
        Transform::from_xyz(0.0, 0.0, 0.0),
        GlobalTransform::default(),
    ));
}

fn setup_graph(
    mut commands: Commands,
    windows: Query<&Window, With<PrimaryWindow>>,
    cfg: Res<Config>
) {
    commands.remove_resource::<Graph>();

    let mut rng = rng();
    let window = windows.single().expect("No window!");
    let (width, height) = (window.width(), window.height());

    // spawn nodes at random positions
    let mut positions = Vec::with_capacity(cfg.node_count);
    for _ in 0..cfg.node_count {
        positions.push(Vec2::new(
            rng.random_range(-width / 2.0..width / 2.0),
            rng.random_range(-height / 2.0..height / 2.0),
        ));
    }

    // connect all nodes
    let mut edges = Vec::new();
    for i in 0..cfg.node_count {
        for j in (i + 1)..cfg.node_count {
            let a = NodeId(i);
            let b = NodeId(j);
            let dist = positions[i].distance(positions[j]);
            edges.push(EdgeData { a, b, distance: dist, pheromone: 1.0 });
        }
    }

    commands.insert_resource(Graph { nodes: positions, edges });
}

fn draw_graph(
    graph: Res<Graph>,
    mut painter: ShapePainter,
) {
    painter.color = Color::WHITE;
    for &pos in graph.nodes.iter() {
        painter.transform.translation = pos.extend(0.0);
        painter.circle(8.0);
    }

    // Determine max pheromone on any edge for normalization
    // let max_pheromones = graph.edges.iter()
    //     .map(|e| e.pheromone)
    //     .fold(0.0_f32, f32::max);
    let max_pheromones = 100.0;

    for edge in graph.edges.iter() {
        if edge.pheromone <= 1.0 {
            continue;
        }

        let pa = graph.nodes[edge.a.0].extend(0.0);
        let pb = graph.nodes[edge.b.0].extend(0.0);

        painter.transform.translation = Vec3::ZERO;
        let alpha = if max_pheromones > 0.0 {
            ((edge.pheromone) / max_pheromones).powf(0.5).clamp(0.0, 1.0)
        } else {
            0.0
        };
        painter.color = Color::srgba(1.0, 0.0, 0.0, alpha);
        painter.thickness = 2.0 * (0.2 + alpha);
        painter.line(pa, pb);
    }
}

fn setup_ants(
    mut commands: Commands,
    graph: Res<Graph>,
    config: Res<Config>,
) {
    let mut rng = rng();
    let rnd_id = rng.random_range(0..graph.nodes.len() - 1);
    let start = NodeId(rnd_id);
    let start_pos = graph.nodes[start.0];

    for _ in 0..config.ant_count {
        commands.spawn((
            Ant {
                current: start,
                target: None,
                visited: vec![start],
                path: Vec::new(),
            },
            Movement {
                direction: Vec2::ZERO,
                speed: 1000.0,
            },
            Transform::from_xyz(start_pos.x, start_pos.y, 100.0),
            GlobalTransform::default(),
        ));
    }
}

fn draw_ants(
    ants_q: Query<&Transform, With<Ant>>,
    mut painter: ShapePainter,
) {
    for tf in ants_q.iter() {
        painter.color = Color::BLACK;
        painter.transform.translation = tf.translation;
        painter.circle(4.0);
    }
}

fn movement_system(
    mut query: Query<(&mut Transform, &Movement)>,
    time: Res<Time>,
) {
    for (mut tf, movement) in query.iter_mut() {
        let velocity = movement.direction.normalize_or_zero() * movement.speed * time.delta_secs();
        tf.translation += velocity.extend(0.0);
    }
}

fn ant_behavior_system(
    mut query: Query<(&mut Movement, &mut Ant, &Transform)>,
    graph: Res<Graph>,
    cfg: Res<Config>
) {
    if graph.nodes.is_empty() {
        return;
    }

    let mut rng = rng();
    let arrive_threshold = 8.0;

    for (mut mv, mut ant, tf) in query.iter_mut() {
        let current_pos = tf.translation.truncate();

        // check if currently moving to a target
        if let Some(target) = ant.target {
            let target_pos = graph.nodes[target.0];
            let delta = target_pos - current_pos;
            let dist = delta.length();

            // TODO: optimize - dont need dist calc without visualization
            // arrived at target
            if dist <= arrive_threshold || !cfg.should_visualize_ants {
                // record visited node and path
                let prev = ant.current;
                ant.visited.push(target);
                ant.path.push((prev, target));
                ant.current = target;
                ant.target = None;
                mv.direction = Vec2::ZERO;
            } else { // keep course
                mv.direction = delta / dist;
            }
        } else { // pick a new random target
            // build list of candidate nodes (excluding current)
            let candidates: Vec<NodeId> = (0..graph.nodes.len())
                .filter_map(|i| {
                    let id = NodeId(i);
                    if id != ant.current && !ant.visited.contains(&id) {
                        Some(id)
                    } else {
                        None
                    }
                })
                .collect();
            // TODO: optimize - dont need to filter candidates if we know that all were visited...
            if candidates.is_empty() {
                continue;
            }
            // calculate path weights: (1/distance)^DISTANCE_POWER * pheromone^PHEROMONE_POWER
            let weights: Vec<f32> = candidates.iter().map(|nid| {
                let d = (graph.nodes[nid.0] - current_pos).length();
                let pheromone = graph.pheromone_between(ant.current, *nid);
                let weight = (1.0_f32 / d).powf(cfg.distance_power) * pheromone.powf(cfg.pheromone_power);
                weight.max(1e-4)
            }).collect();
            // sample one candidate with weighted probability
            let dist_index = WeightedIndex::new(&weights).unwrap();
            let next = candidates[dist_index.sample(&mut rng)];
            // update the movement direction
            let target_pos = graph.nodes[next.0];
            ant.target = Some(next);
            mv.direction = (target_pos - current_pos).normalize_or_zero();
        }
    }
}

fn pheromone_system(
    mut ant_q: Query<(&mut Ant)>,
    mut graph: ResMut<Graph>,
    config: Res<Config>,
) {
    // deposit pheromones based on path's quality
    for (ant)  in ant_q.iter_mut() {
        let total_path_distance: f32 = ant.path.iter().map(|&(x, y)| graph.distance_between(x, y)).sum();
        let deposit = config.pheromone_intensity / total_path_distance;
        for &(x, y) in &ant.path { graph.deposit_pheromone(x, y, deposit); }
    }

    // evaporate pheromones
    const MIN_PHEROMONE: f32 = 1e-4;
    for edge in graph.edges.iter_mut() {
        edge.pheromone *= 1.0 - config.evaporation_rate;
        edge.pheromone = edge.pheromone.clamp(MIN_PHEROMONE, 100.0)
    }
}

fn kill_ants(
    ants_q: Query<Entity, With<Ant>>,
    mut commands: Commands
) {
    for entity in ants_q.iter() {
        commands.entity(entity).despawn();
    }
}

fn stats_system(
    ants_q: Query<&Ant>,
    graph: Res<Graph>,
    mut stats: ResMut<AlgStats>,
) {
    // TODO: optimization - pheromone_system does the same calc. could still be decoupled using an event.
    let mut shortest_path_dist = f32::INFINITY;
    for ant in ants_q.iter() {
        let total_path_distance: f32 = ant.path.iter().map(|&(x, y)| graph.distance_between(x, y)).sum();
        if total_path_distance < shortest_path_dist {
            shortest_path_dist = total_path_distance;
        }
    }
    stats.current_shortest_path_dist = shortest_path_dist;
    if shortest_path_dist < stats.shortest_path_dist {
        stats.shortest_path_dist = shortest_path_dist;
    }
}

fn reset_stats(
    mut stats: ResMut<AlgStats>
) {
    *stats = Default::default();
}

fn run_if_graph_cfg_changed(
    graph: Res<Graph>,
    cfg: Res<Config>
) -> bool {
    graph.nodes.len() != cfg.node_count
}

fn run_if_ants_finished_round(
    ant_q: Query<&Ant>,
    graph: Res<Graph>,
) -> bool {
    let mut is_finished = true;
    for ant in ant_q.iter() {
        if ant.visited.len() != graph.nodes.len() {
            is_finished = false;
        }
    }
    is_finished
}

fn run_if_should_visualize_ants(
    cfg: Res<Config>
) -> bool {
    cfg.should_visualize_ants
}

fn ui_system(
    mut contexts: EguiContexts,
    mut cfg: ResMut<Config>,
    stats: Res<AlgStats>,
) {
    let ctx = contexts.try_ctx_mut();
    match ctx {
        None => { return; }
        Some(ctx) => {
            egui::Window::new("Settings").show(ctx, |ui| {
                ui.add(egui::Slider::new(&mut cfg.node_count, 3..=100).text("Node Count (resets everything!)"));
                ui.add(egui::Slider::new(&mut cfg.ant_count, 1..=50000).text("Ant Count"));
                ui.add(egui::Slider::new(&mut cfg.distance_power, 1.0..=10.0).text("Distance Pow"));
                ui.add(egui::Slider::new(&mut cfg.pheromone_power, 1.0..=10.0).text("Pheromone Pow"));
                ui.add(egui::Slider::new(&mut cfg.pheromone_intensity, 1.0..=10000.0).text("Pheromone Intensity"));
                ui.add(egui::Slider::new(&mut cfg.evaporation_rate, 0.1..=0.99).text("Evaporation Rate"));
                ui.add(egui::Checkbox::new(&mut cfg.should_visualize_ants, "Visualize Ants"));
            });
            egui::Window::new("Stats").show(ctx, |ui| {
                ui.add(egui::Label::new(&format!("Current Shortest Dist: {:.2}", stats.current_shortest_path_dist)));
                ui.add(egui::Label::new(&format!("Shortest Dist: {:.2}", stats.shortest_path_dist)));
            });
        }
    }
}
