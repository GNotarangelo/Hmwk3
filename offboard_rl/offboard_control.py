import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

# Import dei messaggi PX4 standard
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import VehicleLocalPosition

import numpy as np
import time

# --- Configurazione QoS (Quality of Service) per PX4 ---
# I messaggi di controllo (setpoint, comandi) devono essere inviati con policy RTTPS specifiche.
QOS_RCLPY = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1
)

class OffboardControl(Node):
    """
    Nodo ROS 2 per la pianificazione e l'esecuzione di una traiettoria complessa in modalità Offboard su PX4.
    """
    def __init__(self):
        super().__init__('offboard_control')
        self.get_logger().info("Avvio del nodo OffboardControl...")
        
        # --- 1. PUBLISHERS (Invio Dati) ---
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/offboard_control_mode/in', QOS_RCLPY)
        
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/trajectory_setpoint/in', QOS_RCLPY)
        
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/vehicle_command/in', QOS_RCLPY)

        # --- 2. SUBSCRIBERS (Ricezione Dati) ---
        self.local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, '/fmu/vehicle_local_position/out', self.local_position_callback, QOS_RCLPY)

        # --- 3. STATO E CONTROLLO ---
        self.offboard_setpoint_counter = 0  # Contatore per il failsafe
        self.current_local_position = np.array([0.0, 0.0, 0.0]) # [x, y, z] in NED (North, East, Down)
        self.current_yaw = 0.0 # Orientamento corrente

        self.flight_state = "SETUP" # SETUP -> ARMING -> OFFBOARD -> TRAJECTORY

        # --- 4. PARAMETRI DI TRAIETTORIA ---
        self.current_waypoint_index = 0
        
        # DEFINIZIONE DEI WAYPOINT (minimo 7, con velocità continua)
        # Formato: [x (Nord), y (Est), z (Giù, quindi -4 significa 4m di altezza), yaw (orientamento in radianti)]
        # NOTA: PX4 usa il frame NED (North-East-Down).
        self.waypoints = [
            # 1. Decollo (Punto di Riferimento iniziale)
            {'pos': [0.0, 0.0, -4.0], 'yaw': 0.0},
            # 2. Waypoint 1: Inizio del movimento (Velocità > 0)
            {'pos': [5.0, 0.0, -4.0], 'yaw': np.pi/2}, # Gira di 90 gradi verso Est
            # 3. Waypoint 2: Continuazione con curva (Velocità > 0)
            {'pos': [5.0, 5.0, -4.0], 'yaw': np.pi}, # Gira di 90 gradi verso Sud
            # 4. Waypoint 3: Salita (Velocità > 0)
            {'pos': [0.0, 5.0, -6.0], 'yaw': -np.pi/2}, # Gira di 90 gradi verso Ovest e sale a 6m
            # 5. Waypoint 4: Punto alto
            {'pos': [-5.0, 5.0, -6.0], 'yaw': 0.0}, # Va a Ovest
            # 6. Waypoint 5: Discesa
            {'pos': [-5.0, -5.0, -4.0], 'yaw': np.pi}, # Va a Sud-Ovest, scende a 4m
            # 7. Waypoint 6: Ritorno al Centro
            {'pos': [0.0, 0.0, -4.0], 'yaw': 0.0},
            # 8. Waypoint 7 (Frenata/Arrivo)
            {'pos': [0.0, 0.0, -2.0], 'yaw': 0.0}, # Punto di Hover finale
        ]

        # --- 5. LOOP PRINCIPALE ---
        # Il timer esegue il metodo run_loop a 20 Hz (50 ms).
        # Questa è la frequenza minima raccomandata per la modalità Offboard per prevenire il failsafe.
        self.timer = self.create_timer(0.05, self.run_loop)
        self.get_logger().info("Il loop di controllo è stato avviato a 20 Hz.")


    # --- CALLBACKS ---
    def local_position_callback(self, msg):
        """
        Riceve la posizione corrente del drone.
        """
        # La posizione locale è [x, y, z] in metri rispetto al punto di decollo (NED).
        self.current_local_position = np.array([msg.x, msg.y, msg.z])
        # Lo Yaw è l'orientamento attorno all'asse Z (Nord = 0, Est = 90 gradi/pi/2)
        self.current_yaw = msg.heading


    # --- FUNZIONI DI COMANDO ---
    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        """
        Funzione helper per pubblicare i comandi del veicolo.
        """
        msg = VehicleCommand()
        msg.timestamp = int(Clock().now().nanoseconds / 1000)
        msg.command = command
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.vehicle_command_publisher.publish(msg)
        self.get_logger().info(f"Comando inviato: {command}")


    def publish_offboard_control_mode(self):
        """
        Pubblica i messaggi OffboardControlMode per mantenere attiva la modalità.
        Indica a PX4 quali setpoint sta ricevendo (Posizione, Velocità, Atteggiamento, ecc.).
        """
        msg = OffboardControlMode()
        msg.timestamp = int(Clock().now().nanoseconds / 1000)
        # Stiamo inviando Posizione (X, Y, Z) e Orientamento (Yaw)
        msg.position = True
        msg.velocity = False # Non usiamo la velocity in questo esempio, lasciamo che PX4 la calcoli
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        self.offboard_control_mode_publisher.publish(msg)


    def publish_trajectory_setpoint(self, pos, yaw, vel=[np.nan, np.nan, np.nan]):
        """
        Pubblica un setpoint di traiettoria completo.
        """
        msg = TrajectorySetpoint()
        msg.timestamp = int(Clock().now().nanoseconds / 1000)
        
        # 1. Posizione (X, Y, Z)
        msg.position[0] = pos[0] # Nord
        msg.position[1] = pos[1] # Est
        msg.position[2] = pos[2] # Giù (Down)

        # 2. Velocità Desiderata (per traiettoria continua)
        # Se si forniscono NaN, PX4 usa la velocità predefinita o calcolata.
        # Per assicurare un movimento fluido (non-stop), ci affidiamo alla velocità di crociera interna di PX4 
        # e usiamo i setpoint di posizione come punti intermedi (waypoint con velocità > 0 implicita).
        # Solo l'ultimo punto avrà una velocità effettivamente nulla (che è implicita quando ci si ferma a un punto).
        msg.velocity[0] = vel[0]
        msg.velocity[1] = vel[1]
        msg.velocity[2] = vel[2]
        
        # 3. Orientamento (Yaw)
        msg.yaw = yaw
        
        # 4. Orientamento Rate
        msg.yawspeed = 0.0

        self.trajectory_setpoint_publisher.publish(msg)


    # --- LOGICA DI VOLO ---
    def distance_to_waypoint(self, target_pos):
        """
        Calcola la distanza 2D dal waypoint corrente.
        """
        # Distanza in X e Y (evitiamo la Z per la tolleranza orizzontale)
        delta = target_pos[:2] - self.current_local_position[:2]
        return np.linalg.norm(delta)

    
    def run_loop(self):
        """
        Loop principale eseguito a 20 Hz. Gestisce la transizione di stato e l'invio dei setpoint.
        """
        timestamp = int(Clock().now().nanoseconds / 1000)
        
        # 1. Manteniamo il flusso continuo per il failsafe Offboard
        self.publish_offboard_control_mode()

        # --- A. SETUP: Pre-Arming (Invia un setpoint iniziale) ---
        if self.flight_state == "SETUP":
            # Invia un setpoint statico iniziale (altrimenti PX4 rifiuta la modalità Offboard)
            target = self.waypoints[0]
            self.publish_trajectory_setpoint(target['pos'], target['yaw'])
            
            # Controlla se sono stati inviati abbastanza setpoint per attivare Offboard (> 100)
            if self.offboard_setpoint_counter >= 100:
                self.flight_state = "ARMING"
                self.get_logger().info("Setup completo. Pronto per Arming.")

        # --- B. ARMING: Armare il drone ---
        elif self.flight_state == "ARMING":
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
            time.sleep(1) # Aspetta un secondo per l'armamento
            self.flight_state = "OFFBOARD"
            self.get_logger().info("Drone Armato. Passaggio a Offboard...")
        
        # --- C. OFFBOARD: Attivazione Offboard e Decollo (Waypoint 0) ---
        elif self.flight_state == "OFFBOARD":
            # Invia il comando per passare alla modalità Offboard
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0) # param2=6.0 è MAV_MODE_FLAG_CUSTOM_MODE_OFFBOARD
            
            target = self.waypoints[0]
            self.publish_trajectory_setpoint(target['pos'], target['yaw'])

            # Controlla se il drone è arrivato al punto di decollo
            if self.distance_to_waypoint(target['pos']) < 0.3 and abs(self.current_local_position[2] - target['pos'][2]) < 0.2:
                self.get_logger().info("Decollo completato. Inizio traiettoria...")
                self.current_waypoint_index = 1
                self.flight_state = "TRAJECTORY"
        
        # --- D. TRAJECTORY: Navigazione continua tra i waypoint ---
        elif self.flight_state == "TRAJECTORY":
            if self.current_waypoint_index < len(self.waypoints):
                target = self.waypoints[self.current_waypoint_index]
                
                # Invia il setpoint di Posizione e Orientamento
                self.publish_trajectory_setpoint(target['pos'], target['yaw'])
                self.get_logger().info(f"Targeting Waypoint {self.current_waypoint_index} -> Pos: {target['pos'][0]:.1f}, {target['pos'][1]:.1f}, {target['pos'][2]:.1f}")
                
                # Criterio di transizione: se siamo vicini al waypoint corrente
                if self.distance_to_waypoint(target['pos']) < 1.0: # Tolleranza di 1 metro
                    self.current_waypoint_index += 1
                    # Se è l'ultimo waypoint (l'arrivo), passiamo alla fase di atterraggio.
                    if self.current_waypoint_index == len(self.waypoints) - 1:
                         self.flight_state = "LANDING_PREP"
                         self.get_logger().info("Ultimo Waypoint raggiunto. Preparazione per Landing...")
                    else:
                        self.get_logger().info(f"Waypoint {self.current_waypoint_index-1} raggiunto. Prossimo...")

            else:
                 self.flight_state = "DISARMING"

        # --- E. LANDING_PREP: Atterraggio (Ultimo Waypoint) ---
        elif self.flight_state == "LANDING_PREP":
            target = self.waypoints[-1] # L'ultimo punto, dove si ferma e atterra
            
            # Invia il setpoint finale, dove ci si aspetta che la velocità si azzeri.
            # Qui potresti anche passare alla modalità 'Land' (VehicleCommand.VEHICLE_CMD_NAV_LAND)
            # ma per l'esercizio, continuiamo a inviare il setpoint statico.
            self.publish_trajectory_setpoint(target['pos'], target['yaw'])
            
            # Se siamo al punto finale e sufficientemente bassi, disarmare.
            if self.distance_to_waypoint(target['pos']) < 0.1 and abs(self.current_local_position[2]) > -0.5:
                self.flight_state = "DISARMING"
                self.get_logger().info("Atterraggio virtuale completato.")

        # --- F. DISARMING ---
        elif self.flight_state == "DISARMING":
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
            self.flight_state = "DONE"
            self.get_logger().info("Esercizio completato. Drone Disarmato.")
        
        # --- G. DONE ---
        elif self.flight_state == "DONE":
            pass # Non fa nulla
        
        # Incrementa il contatore per il setup iniziale
        self.offboard_setpoint_counter += 1


def main(args=None):
    rclpy.init(args=args)
    offboard_control = OffboardControl()
    
    # Esegue il nodo fino all'interruzione (Ctrl+C)
    try:
        rclpy.spin(offboard_control)
    except KeyboardInterrupt:
        pass

    # Pulizia
    offboard_control.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
