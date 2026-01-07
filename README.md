# Drone-Swarm-Simulation
import numpy as np
import socket
import threading
import time
import json
from PyFlyt.core import Aviary
import queue
import random
import pybullet as p
import math

class MovingTarget:
    def __init__(self, target_id, initial_position, velocity, detection_range=3.0):
        self.target_id = target_id
        self.position = np.array(initial_position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.marker_id = None
        self.detection_range = detection_range
        self.discovered = False
        self.discovered_by = []
        self.visible = False  # Track if marker is visible
        
    def update_position(self, dt):
        """Update target position based on velocity with boundary constraints"""
        new_pos = self.position + self.velocity * dt
        
        # Boundary constraints (keep targets in reasonable area)
        bounds = 15.0
        for i in range(2):  # x and y only
            if abs(new_pos[i]) > bounds:
                self.velocity[i] *= -1  # Bounce off boundaries
                new_pos[i] = np.clip(new_pos[i], -bounds, bounds)
        
        # Keep targets at ground level (below drones)
        new_pos[2] = 0.3  # Ground level
        self.position = new_pos
    
    def is_detected_by(self, drone_position):
        """Check if target can be detected by drone at given position"""
        distance = np.linalg.norm(drone_position - self.position)
        return distance <= self.detection_range
    
    def get_position(self):
        return self.position.copy()
    
    def get_velocity(self):
        return self.velocity.copy()

class SearchPattern:
    def __init__(self, drone_id, search_area_bounds=15.0):
        self.drone_id = drone_id
        self.search_area_bounds = search_area_bounds
        self.pattern_type = "spiral"
        self.search_center = np.zeros(3)
        self.search_radius = 2.0
        self.spiral_angle = 0.0
        self.spiral_radius = 1.0
        self.max_spiral_radius = 8.0
        self.grid_size = 3.0
        self.grid_position = [0, 0]
        self.waypoint_reached_threshold = 1.0
        
    def get_search_waypoint(self, current_position, dt):
        """Generate next waypoint based on search pattern"""
        if self.pattern_type == "spiral":
            return self._spiral_search(current_position, dt)
        elif self.pattern_type == "grid":
            return self._grid_search(current_position, dt)
        else:
            return self._random_search(current_position, dt)
    
    def _spiral_search(self, current_position, dt):
        """Spiral search pattern expanding outward"""
        waypoint = self.search_center.copy()
        waypoint[0] += self.spiral_radius * np.cos(self.spiral_angle)
        waypoint[1] += self.spiral_radius * np.sin(self.spiral_angle)
        waypoint[2] = current_position[2]
        
        self.spiral_angle += 0.1
        if self.spiral_angle > 2 * np.pi:
            self.spiral_angle = 0.0
            self.spiral_radius = min(self.spiral_radius + 0.5, self.max_spiral_radius)
        
        return waypoint
    
    def _grid_search(self, current_position, dt):
        """Grid search pattern"""
        waypoint = self.search_center.copy()
        waypoint[0] += self.grid_position[0] * self.grid_size
        waypoint[1] += self.grid_position[1] * self.grid_size
        waypoint[2] = current_position[2]
        
        if np.linalg.norm(current_position - waypoint) < self.waypoint_reached_threshold:
            self.grid_position[0] += 1
            if self.grid_position[0] > 4:
                self.grid_position[0] = -4
                self.grid_position[1] += 1
            if self.grid_position[1] > 4:
                self.grid_position[1] = -4
        
        return waypoint
    
    def _random_search(self, current_position, dt):
        """Random waypoint search"""
        if not hasattr(self, 'current_waypoint') or \
           np.linalg.norm(current_position - self.current_waypoint) < self.waypoint_reached_threshold:
            self.current_waypoint = np.array([
                random.uniform(-self.search_area_bounds, self.search_area_bounds),
                random.uniform(-self.search_area_bounds, self.search_area_bounds),
                current_position[2]
            ])
        
        return self.current_waypoint

class DroneSwarmCommunication:
    def __init__(self, drone_id, num_drones=5, base_port=12000, communication_range=10.0):
        self.drone_id = drone_id
        self.num_drones = num_drones
        self.base_port = base_port
        self.communication_range = communication_range
        self.alpha = 0.02
        
        # Communication setup
        self.my_port = base_port + drone_id
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(('localhost', self.my_port))
        self.socket.settimeout(0.1)
        
        # Drone state
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        
        # Target information shared between drones
        self.discovered_targets = {}
        
        # Obstacle information
        self.obstacles = []
        
        # Neighbor information
        self.neighbor_states = {}
        
        # Communication thread
        self.running = True
        self.comm_thread = threading.Thread(target=self._communication_loop, daemon=True)
        self.comm_thread.start()
        
        print(f"Drone {drone_id} initialized on port {self.my_port}")
    
    def set_obstacles(self, obstacles):
        """Set obstacle positions for this drone"""
        self.obstacles = obstacles
    
    def add_discovered_target(self, target_id, target_position, target_velocity):
        """Add a newly discovered target to the shared database"""
        self.discovered_targets[target_id] = {
            'position': np.array(target_position),
            'velocity': np.array(target_velocity),
            'last_seen': time.time(),
            'discovered_by': self.drone_id,
            'tracking_drone': self.drone_id  # Immediately assign to discoverer
        }
        print(f"Drone {self.drone_id} discovered and assigned to track target {target_id}")
    
    def update_target_position(self, target_id, target_position, target_velocity):
        """Update position of a known target"""
        if target_id in self.discovered_targets:
            self.discovered_targets[target_id]['position'] = np.array(target_position)
            self.discovered_targets[target_id]['velocity'] = np.array(target_velocity)
            self.discovered_targets[target_id]['last_seen'] = time.time()
    
    def _communication_loop(self):
        """Continuous communication loop for receiving messages"""
        while self.running:
            try:
                data, addr = self.socket.recvfrom(2048)
                message = json.loads(data.decode())
                
                sender_id = message['drone_id']
                if sender_id != self.drone_id:
                    # Update neighbor state
                    self.neighbor_states[sender_id] = {
                        'position': np.array(message['position']),
                        'velocity': np.array(message['velocity']),
                        'timestamp': time.time()
                    }
                    
                    # Update discovered targets from neighbors
                    if 'discovered_targets' in message:
                        for target_id_str, target_info in message['discovered_targets'].items():
                            target_id = int(target_id_str)
                            
                            if target_id not in self.discovered_targets or \
                               target_info['last_seen'] > self.discovered_targets[target_id]['last_seen']:
                                self.discovered_targets[target_id] = {
                                    'position': np.array(target_info['position']),
                                    'velocity': np.array(target_info['velocity']),
                                    'last_seen': target_info['last_seen'],
                                    'discovered_by': target_info['discovered_by'],
                                    'tracking_drone': target_info.get('tracking_drone', None)
                                }
                                
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"Drone {self.drone_id} communication error: {e}")
    
    def broadcast_state(self):
        """Broadcast current state and discovered targets to all other drones"""
        targets_data = {}
        for target_id, target_info in self.discovered_targets.items():
            targets_data[str(target_id)] = {
                'position': target_info['position'].tolist(),
                'velocity': target_info['velocity'].tolist(),
                'last_seen': target_info['last_seen'],
                'discovered_by': target_info['discovered_by'],
                'tracking_drone': target_info.get('tracking_drone', None)
            }
        
        message = {
            'drone_id': self.drone_id,
            'position': self.position.tolist(),
            'velocity': self.velocity.tolist(),
            'timestamp': time.time(),
            'discovered_targets': targets_data
        }
        
        message_data = json.dumps(message).encode()
        
        for other_drone_id in range(self.num_drones):
            if other_drone_id != self.drone_id:
                try:
                    target_port = self.base_port + other_drone_id
                    self.socket.sendto(message_data, ('localhost', target_port))
                except Exception as e:
                    pass
    
    def get_neighbors(self):
        """Get current neighbors based on communication range"""
        neighbors = []
        current_time = time.time()
        
        for drone_id, state in self.neighbor_states.items():
            if current_time - state['timestamp'] > 1.0:
                continue
                
            distance = np.linalg.norm(self.position - state['position'])
            if distance <= self.communication_range:
                neighbors.append(drone_id)
        
        return neighbors
    
    def update_consensus_velocity(self):
        """Update velocity using consensus protocol"""
        neighbors = self.get_neighbors()
        
        if not neighbors:
            return self.velocity
        
        velocity_sum = np.zeros(3)
        for neighbor_id in neighbors:
            neighbor_velocity = self.neighbor_states[neighbor_id]['velocity']
            velocity_sum += (neighbor_velocity - self.velocity)
        
        consensus_velocity = self.velocity + (self.alpha / len(neighbors)) * velocity_sum
        
        return consensus_velocity
    
    def check_obstacle_avoidance(self, desired_velocity, obstacle_avoidance_distance=2.5):
        """Enhanced 3D obstacle avoidance"""
        avoidance_velocity = np.zeros(3)
        
        for obstacle_pos, obstacle_radius in self.obstacles:
            direction = self.position - obstacle_pos
            distance = np.linalg.norm(direction)
            
            critical_distance = obstacle_radius + obstacle_avoidance_distance
            
            if distance < critical_distance and distance > 0.01:
                normalized_distance = max(0.01, (distance - obstacle_radius) / obstacle_avoidance_distance)
                repulsive_strength = 4.5 * (1.0 / normalized_distance - 1.0)
                repulsive_strength = max(repulsive_strength, 2.0)
                
                repulsive_direction = direction / distance
                repulsive_force = repulsive_direction * repulsive_strength
                
                avoidance_velocity += repulsive_force
                
                height_diff = self.position[2] - obstacle_pos[2]
                if abs(height_diff) < obstacle_radius + 1.0:
                    if height_diff >= 0:
                        avoidance_velocity[2] += 2.0
                    else:
                        avoidance_velocity[2] -= 1.0
        
        return desired_velocity + avoidance_velocity
    
    def check_collision_avoidance(self, desired_velocity, min_distance=1.8):
        """Enhanced collision avoidance with other drones"""
        neighbors = self.get_neighbors()
        avoidance_velocity = np.zeros(3)
        
        for neighbor_id in neighbors:
            neighbor_pos = self.neighbor_states[neighbor_id]['position']
            neighbor_vel = self.neighbor_states[neighbor_id]['velocity']
            
            direction = self.position - neighbor_pos
            distance = np.linalg.norm(direction)
            
            if distance < min_distance and distance > 0.01:
                priority_factor = 1.0
                if self.drone_id > neighbor_id:
                    priority_factor = 1.3
                
                avoidance_force = (direction / distance) * (min_distance - distance) * 3.0 * priority_factor
                avoidance_velocity += avoidance_force
            
            relative_velocity = self.velocity - neighbor_vel
            if np.linalg.norm(relative_velocity) > 0.1:
                time_to_closest = -np.dot(direction, relative_velocity) / np.dot(relative_velocity, relative_velocity)
                
                if 0 < time_to_closest < 2.0:
                    closest_distance = np.linalg.norm(direction + relative_velocity * time_to_closest)
                    
                    if closest_distance < min_distance:
                        future_separation = direction + relative_velocity * time_to_closest
                        if np.linalg.norm(future_separation) > 0.01:
                            predictive_force = (future_separation / np.linalg.norm(future_separation)) * 0.9
                            avoidance_velocity += predictive_force
        
        return desired_velocity + avoidance_velocity
    
    def get_assigned_target(self):
        """Get target assigned to this drone"""
        for target_id, target_info in self.discovered_targets.items():
            if target_info.get('tracking_drone', None) == self.drone_id:
                return (target_id, target_info)
        return None
    
    def update_state(self, position, velocity):
        """Update drone's current state"""
        self.position = np.array(position)
        self.velocity = np.array(velocity)
    
    def close(self):
        """Clean shutdown"""
        self.running = False
        self.socket.close()
        if self.comm_thread.is_alive():
            self.comm_thread.join(timeout=1.0)

class DroneSwarmSearchSimulation:
    def __init__(self, num_drones=5, num_targets=3):
        self.num_drones = num_drones
        self.num_targets = num_targets
        self.dt = 1.0/240.0
        
        # Define starting positions at different heights
        self.start_pos = self._generate_start_positions(num_drones)
        self.start_orn = np.zeros((num_drones, 3))
        
        # Generate 3D obstacles at various heights
        self.obstacles = self._generate_3d_obstacles()
        
        # Initialize PyFlyt environment
        self.env = Aviary(
            start_pos=self.start_pos,
            start_orn=self.start_orn,
            render=True,
            drone_type="quadx"
        )
        
        self.env.set_mode(6)
        
        # Add visible obstacles
        self.obstacle_ids = self._add_visible_obstacles()
        
        # Initialize unknown moving targets (ground level)
        self.moving_targets = self._initialize_unknown_targets()
        
        # Initialize communication system
        self.drone_comms = []
        for i in range(num_drones):
            comm = DroneSwarmCommunication(drone_id=i, num_drones=num_drones)
            comm.set_obstacles(self.obstacles)
            self.drone_comms.append(comm)
        
        # Initialize search patterns for each drone
        self.search_patterns = []
        search_types = ["spiral", "grid", "random"]
        for i in range(num_drones):
            pattern = SearchPattern(i)
            pattern.pattern_type = search_types[i % len(search_types)]
            angle = 2 * np.pi * i / num_drones
            pattern.search_center = np.array([5 * np.cos(angle), 5 * np.sin(angle), 0])
            self.search_patterns.append(pattern)
        
        print(f"Initialized search simulation with {num_drones} drones and {num_targets} hidden targets")
        print(f"Generated {len(self.obstacles)} 3D obstacles")
        
        # Statistics
        self.targets_discovered = 0
        self.targets_tracked = 0
        self.discovery_times = []
    
    def _generate_start_positions(self, num_drones):
        """Generate start positions at different heights"""
        positions = []
        base_height = 2.5
        
        for i in range(num_drones):
            angle = 2 * np.pi * i / num_drones
            radius = 3.0
            height = base_height + (i % 3) * 1.0
            
            pos = [
                radius * np.cos(angle),
                radius * np.sin(angle),
                height
            ]
            positions.append(pos)
        
        return np.array(positions)
    
    def _generate_3d_obstacles(self):
        """Generate obstacles at various heights affecting drone paths"""
        obstacles = []
        
        ground_obstacles = [
            ([4.0, 2.0, 0.8], 0.6),
            ([-3.0, 4.0, 1.2], 0.5),
            ([6.0, -2.0, 0.9], 0.4)
        ]
        
        mid_level_obstacles = [
            ([2.0, 3.0, 2.8], 0.7),
            ([-4.0, 1.0, 3.2], 0.6),
            ([5.0, -3.0, 2.5], 0.5),
            ([0.0, -4.0, 3.5], 0.6),
            ([-2.0, -2.0, 4.0], 0.4)
        ]
        
        high_level_obstacles = [
            ([3.0, -1.0, 4.5], 0.5),
            ([-1.0, 3.0, 5.0], 0.4),
            ([0.0, 2.0, 4.8], 0.6)
        ]
        
        obstacles.extend(ground_obstacles)
        obstacles.extend(mid_level_obstacles)
        obstacles.extend(high_level_obstacles)
        
        for i in range(len(obstacles)):
            obstacles[i] = (np.array(obstacles[i][0]), obstacles[i][1])
        
        return obstacles
    
    def _initialize_unknown_targets(self):
        """Initialize moving targets at ground level (unknown to drones)"""
        targets = []
        
        for i in range(self.num_targets):
            while True:
                x = random.uniform(-12.0, 12.0)
                y = random.uniform(-12.0, 12.0)
                z = 0.3
                position = [x, y, z]
                
                min_distance_to_drones = min([
                    np.linalg.norm(np.array(position) - start_pos) 
                    for start_pos in self.start_pos
                ])
                
                if min_distance_to_drones > 4.0:
                    break
            
            speed = random.uniform(0.8, 1.8)
            direction = random.uniform(0, 2 * np.pi)
            velocity = [speed * np.cos(direction), speed * np.sin(direction), 0.0]
            
            target = MovingTarget(
                target_id=i,
                initial_position=position,
                velocity=velocity,
                detection_range=3.0
            )
            targets.append(target)
            
            print(f"Target {i} placed at [{x:.1f}, {y:.1f}, {z:.1f}] with velocity [{velocity[0]:.1f}, {velocity[1]:.1f}, {velocity[2]:.1f}] (HIDDEN from drones)")
        
        return targets
    
    def _add_visible_obstacles(self):
        """Add visible obstacles to the simulation"""
        obstacle_ids = []
        
        try:
            if hasattr(self.env, '_p'):
                physics_client = self.env._p
            elif hasattr(self.env, 'BC'):
                physics_client = self.env.BC
            else:
                physics_client = p
        except:
            physics_client = p
        
        for i, (obstacle_pos, obstacle_radius) in enumerate(self.obstacles):
            try:
                if obstacle_pos[2] < 2.0:
                    color = [0.6, 0.3, 0.1, 0.8]
                elif obstacle_pos[2] < 4.0:
                    color = [0.8, 0.1, 0.1, 0.9]
                else:
                    color = [0.1, 0.1, 0.8, 0.9]
                
                visual_shape_id = physics_client.createVisualShape(
                    shapeType=p.GEOM_SPHERE,
                    radius=obstacle_radius,
                    rgbaColor=color
                )
                
                collision_shape_id = physics_client.createCollisionShape(
                    shapeType=p.GEOM_SPHERE,
                    radius=obstacle_radius
                )
                
                obstacle_id = physics_client.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=collision_shape_id,
                    baseVisualShapeIndex=visual_shape_id,
                    basePosition=obstacle_pos.tolist()
                )
                
                obstacle_ids.append(obstacle_id)
                
            except Exception as e:
                print(f"Failed to create obstacle {i+1}: {e}")
        
        return obstacle_ids
    
    def _create_target_marker(self, target_id):
        """Create visible target marker immediately when discovered"""
        try:
            if hasattr(self.env, '_p'):
                physics_client = self.env._p
            elif hasattr(self.env, 'BC'):
                physics_client = self.env.BC
            else:
                physics_client = p
        except:
            physics_client = p
        
        target = self.moving_targets[target_id]
        
        try:
            visual_shape_id = physics_client.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=0.3,
                rgbaColor=[0.1, 0.9, 0.1, 0.9]
            )
            
            marker_id = physics_client.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=visual_shape_id,
                basePosition=target.get_position().tolist()
            )
            
            target.marker_id = marker_id
            target.visible = True
            print(f"GREEN BALL created for Target {target_id} at position {target.get_position()}")
            
        except Exception as e:
            print(f"Failed to create target marker {target_id}: {e}")
    
    def _update_target_markers(self):
        """Update visual markers for discovered targets"""
        try:
            if hasattr(self.env, '_p'):
                physics_client = self.env._p
            elif hasattr(self.env, 'BC'):
                physics_client = self.env.BC
            else:
                physics_client = p
        except:
            physics_client = p
        
        for target in self.moving_targets:
            if target.visible and target.marker_id is not None:
                try:
                    physics_client.resetBasePositionAndOrientation(
                        target.marker_id,
                        target.get_position().tolist(),
                        [0, 0, 0, 1]
                    )
                except Exception as e:
                    pass
    
    def run_simulation(self, max_steps=8000):
        """Run the search simulation"""
        print("\nStarting drone swarm TARGET SEARCH simulation...")
        print("Targets become GREEN BALLS when discovered!")
        
        start_time = time.time()
        last_status_print = time.time()
        
        for step in range(max_steps):
            # Update moving targets
            for target in self.moving_targets:
                target.update_position(self.dt)
            
            # Get current drone states
            drone_positions = []
            
            for i in range(self.num_drones):
                state = self.env.state(i)
                position = np.array(state[3])
                drone_positions.append(position)
                
                if step > 0:
                    self.drone_comms[i].update_state(position, self.drone_comms[i].velocity)
                else:
                    self.drone_comms[i].update_state(position, np.zeros(3))
            
            # Check for target discoveries
            self._check_target_discoveries(drone_positions)
            
            # Process each drone's behavior
            for i in range(self.num_drones):
                self.drone_comms[i].broadcast_state()
                
                current_pos = drone_positions[i]
                assigned_target = self.drone_comms[i].get_assigned_target()
                
                if assigned_target is not None:
                    # TRACKING MODE - Follow the green ball
                    target_id, target_info = assigned_target
                    actual_target = self.moving_targets[target_id]
                    
                    # Get current target position
                    current_target_pos = actual_target.get_position()
                    current_target_vel = actual_target.get_velocity()
                    
                    # Update target info continuously
                    if actual_target.is_detected_by(current_pos):
                        self.drone_comms[i].update_target_position(
                            target_id, current_target_pos, current_target_vel
                        )
                    
                    # Calculate distance to target
                    direction_to_target = current_target_pos - current_pos
                    distance_to_target = np.linalg.norm(direction_to_target)
                    
                    # IMPROVED FOLLOWING BEHAVIOR
                    if distance_to_target > 0.01:
                        if distance_to_target > 3.0:
                            # Far from target - move fast toward it
                            desired_velocity = (direction_to_target / distance_to_target) * 3.0
                        elif distance_to_target > 1.5:
                            # Medium distance - intercept trajectory
                            time_to_intercept = distance_to_target / 2.5
                            predicted_pos = current_target_pos + current_target_vel * time_to_intercept
                            intercept_dir = predicted_pos - current_pos
                            if np.linalg.norm(intercept_dir) > 0.01:
                                desired_velocity = (intercept_dir / np.linalg.norm(intercept_dir)) * 2.5
                            else:
                                desired_velocity = current_target_vel.copy()
                        else:
                            # Close to target - follow closely
                            desired_velocity = current_target_vel + (direction_to_target / distance_to_target) * 1.0
                    else:
                        # On top of target - match velocity
                        desired_velocity = current_target_vel.copy()
                    
                    print(f"Drone {i} tracking Target {target_id}: distance={distance_to_target:.2f}m")
                
                else:
                    # SEARCH MODE
                    search_waypoint = self.search_patterns[i].get_search_waypoint(current_pos, self.dt)
                    direction_to_waypoint = search_waypoint - current_pos
                    distance_to_waypoint = np.linalg.norm(direction_to_waypoint)
                    
                    if distance_to_waypoint > 0.01:
                        desired_velocity = (direction_to_waypoint / distance_to_waypoint) * 1.5
                    else:
                        desired_velocity = np.zeros(3)
                
                # Apply safety measures
                consensus_velocity = self.drone_comms[i].update_consensus_velocity()
                if assigned_target is not None:
                    blended_velocity = 0.98 * desired_velocity + 0.02 * consensus_velocity
                else:
                    blended_velocity = 0.7 * desired_velocity + 0.3 * consensus_velocity
                
                obstacle_avoided_velocity = self.drone_comms[i].check_obstacle_avoidance(blended_velocity)
                safe_velocity = self.drone_comms[i].check_collision_avoidance(obstacle_avoided_velocity)
                
                # Limit velocity
                max_velocity = 3.5 if assigned_target is not None else 2.0
                velocity_magnitude = np.linalg.norm(safe_velocity)
                if velocity_magnitude > max_velocity:
                    safe_velocity = (safe_velocity / velocity_magnitude) * max_velocity
                
                self.drone_comms[i].velocity = safe_velocity
                
                # Send command to PyFlyt
                vel_command = np.array([safe_velocity[0], safe_velocity[1], safe_velocity[2], 0.0])
                self.env.set_setpoint(i, vel_command)
            
            # Update target markers
            self._update_target_markers()
            
            # Step simulation
            self.env.step()
            
            # Print status every 5 seconds
            current_time = time.time()
            if current_time - last_status_print >= 5.0:
                self._print_search_status(step, current_time - start_time)
                last_status_print = current_time
            
            # Check if all targets discovered
            if self.targets_discovered >= self.num_targets:
                print(f"\nALL TARGETS DISCOVERED! Continuing tracking...")
        
        print("\nSearch simulation completed!")
        self.cleanup()
    
    def _check_target_discoveries(self, drone_positions):
        """Check if any drones discovered new targets"""
        for drone_id, drone_pos in enumerate(drone_positions):
            for target_id, target in enumerate(self.moving_targets):
                if not target.discovered and target.is_detected_by(drone_pos):
                    # Target discovered!
                    target.discovered = True
                    target.discovered_by.append(drone_id)
                    
                    # Create green ball marker IMMEDIATELY
                    self._create_target_marker(target_id)
                    
                    # Add to drone's discovered targets database
                    self.drone_comms[drone_id].add_discovered_target(
                        target_id, 
                        target.get_position(), 
                        target.get_velocity()
                    )
                    
                    # Update statistics
                    self.targets_discovered += 1
                    self.discovery_times.append(time.time())
                    
                    print(f"🎯 TARGET {target_id} DISCOVERED by Drone {drone_id}! GREEN BALL APPEARS!")
                    print(f"   Position: {target.get_position()}")
                    print(f"   Targets found: {self.targets_discovered}/{self.num_targets}")
    
    def _print_search_status(self, step, elapsed_time):
        """Print current search status"""
        print(f"\n--- SEARCH STATUS (Step {step}, Time: {elapsed_time:.1f}s) ---")
        print(f"Targets discovered: {self.targets_discovered}/{self.num_targets}")
        
        # Count drones in search vs tracking mode
        search_drones = 0
        tracking_drones = 0
        
        for i, comm in enumerate(self.drone_comms):
            assigned_target = comm.get_assigned_target()
            if assigned_target is not None:
                tracking_drones += 1
                target_id = assigned_target[0]
                target_pos = self.moving_targets[target_id].get_position()
                drone_pos = comm.position
                distance = np.linalg.norm(drone_pos - target_pos)
                print(f"  Drone {i}: TRACKING Target {target_id} (distance: {distance:.1f}m)")
            else:
                search_drones += 1
                pattern_type = self.search_patterns[i].pattern_type
                print(f"  Drone {i}: SEARCHING ({pattern_type} pattern)")
        
        print(f"Drones searching: {search_drones}, Drones tracking: {tracking_drones}")
        
        # Print discovered targets info
        if self.targets_discovered > 0:
            print("Discovered targets (GREEN BALLS):")
            for target_id, target in enumerate(self.moving_targets):
                if target.discovered:
                    pos = target.get_position()
                    vel = target.get_velocity()
                    print(f"  Target {target_id}: pos=[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}], "
                          f"vel=[{vel[0]:.1f}, {vel[1]:.1f}, {vel[2]:.1f}] - VISIBLE!")
    
    def cleanup(self):
        """Clean shutdown"""
        print("Shutting down search simulation...")
        
        for comm in self.drone_comms:
            comm.close()
        
        try:
            if hasattr(self.env, '_p'):
                physics_client = self.env._p
            elif hasattr(self.env, 'BC'):
                physics_client = self.env.BC
            else:
                physics_client = p
                
            # Remove obstacles
            for obstacle_id in self.obstacle_ids:
                physics_client.removeBody(obstacle_id)
            
            # Remove target markers
            for target in self.moving_targets:
                if target.marker_id is not None:
                    physics_client.removeBody(target.marker_id)
                
        except Exception as e:
            print(f"Error during cleanup: {e}")
        
        try:
            if hasattr(self.env, 'close'):
                self.env.close()
        except:
            del self.env

def main():
    """Main function to run the drone swarm search simulation"""
    print("="*70)
    print("           DRONE SWARM TARGET SEARCH SIMULATION")
    print("="*70)
    print("\nMISSION: Find hidden targets - they become GREEN BALLS when discovered!")
    print("GOAL: Drones must then FOLLOW the green balls closely!")
    
    print("\nSimulation Parameters:")
    num_drones = 5
    num_targets = 1
    print(f"- Number of search drones: {num_drones}")
    print(f"- Number of hidden targets: {num_targets}")
    print(f"- Target detection range: 3.0 meters")
    print(f"- When discovered: Target becomes GREEN BALL")
    print(f"- Drone behavior: Search → Discover → Follow green ball")
    
    print("\nStarting simulation in 3 seconds...")
    time.sleep(3)
    
    simulation = DroneSwarmSearchSimulation(num_drones=num_drones, num_targets=num_targets)
    
    try:
        simulation.run_simulation(max_steps=8000)
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"Simulation error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation.cleanup()
    
    print("\nSimulation ended!")

if __name__ == "__main__":
    main()
