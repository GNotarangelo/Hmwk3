#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include <cmath>
#include <px4_msgs/msg/vehicle_local_position.hpp>
#include <px4_msgs/msg/vehicle_attitude.hpp>
#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <Eigen/Dense>
#include <chrono>

using namespace std::chrono_literals;
using namespace px4_msgs::msg;
using Eigen::Vector3d;

// Definizione della struttura Waypoint, simile al dizionario Python
struct Waypoint {
    Vector3d position; // [x, y, z] in NED
    double yaw;        // Orientamento in radianti
};

// Enum per la Macchina a Stati (traduzione dello stato Python)
enum class FlightState {
    SETUP,
    ARMING,
    TRAJECTORY,
    LANDING_PREP,
    DONE
};

// --- Configurazione QoS (Quality of Service) ---
rclcpp::QoS qos_profile_command() {
    return rclcpp::QoS(rclcpp::QoSInitialization(RMW_QOS_POLICY_HISTORY_KEEP_LAST, 1),
                       rmw_qos_profile_t{
                           RMW_QOS_POLICY_HISTORY_KEEP_LAST,
                           1,
                           RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
                           RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL,
                           RMW_QOS_DEADLINE_DEFAULT,
                           RMW_QOS_LIFESPAN_DEFAULT,
                           RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
                           RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
                           false
                       });
}

rclcpp::QoS qos_profile_telemetry() {
    return rclcpp::QoS(rclcpp::QoSInitialization(RMW_QOS_POLICY_HISTORY_KEEP_LAST, 5),
                       rmw_qos_profile_t{
                           RMW_QOS_POLICY_HISTORY_KEEP_LAST,
                           5,
                           RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
                           RMW_QOS_POLICY_DURABILITY_VOLATILE,
                           RMW_QOS_DEADLINE_DEFAULT,
                           RMW_QOS_LIFESPAN_DEFAULT,
                           RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
                           RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
                           false
                       });
}


class OffboardWaypointController : public rclcpp::Node
{
public:
    OffboardWaypointController() : Node("offboard_waypoint_controller_cpp")
    {
        RCLCPP_INFO(this->get_logger(), "Avvio del nodo OffboardWaypointController C++...");

        // 1. PUBLISHERS
        offboard_control_mode_publisher_ = this->create_publisher<OffboardControlMode>("/fmu/in/offboard_control_mode", qos_profile_command());
        trajectory_setpoint_publisher_ = this->create_publisher<TrajectorySetpoint>("/fmu/in/trajectory_setpoint", qos_profile_command());
        vehicle_command_publisher_ = this->create_publisher<VehicleCommand>("/fmu/in/vehicle_command", qos_profile_command());

        // 2. SUBSCRIBERS
        local_position_subscription_ = this->create_subscription<VehicleLocalPosition>("/fmu/out/vehicle_local_position", qos_profile_telemetry(),
            std::bind(&OffboardWaypointController::vehicle_local_position_callback, this, std::placeholders::_1));
        
        // 3. TIMER (20 Hz)
        timer_ = this->create_wall_timer(50ms, std::bind(&OffboardWaypointController::run_loop, this));
    }

private:
    rclcpp::Subscription<VehicleLocalPosition>::SharedPtr local_position_subscription_;
    rclcpp::Publisher<OffboardControlMode>::SharedPtr offboard_control_mode_publisher_;
    rclcpp::Publisher<TrajectorySetpoint>::SharedPtr trajectory_setpoint_publisher_;
    rclcpp::Publisher<VehicleCommand>::SharedPtr vehicle_command_publisher_;
    rclcpp::TimerBase::SharedPtr timer_;

    // Variabili di Stato
    Vector3d current_position_{0.0, 0.0, 0.0};
    double current_yaw_ {0.0};
    uint64_t last_position_timestamp_ = 0; // Tempo in nanosecondi
    double offboard_setpoint_counter_ = 0;
    size_t current_waypoint_index_ = 0;
    bool is_landing_commanded_ = false;
    FlightState flight_state_ = FlightState::SETUP;

    // DEFINIZIONE DEI WAYPOINT (tradotti dalla versione Python)
    const std::vector<Waypoint> waypoints_ = {
        // [x, y, z] in NED. z negativo è altezza positiva.
        // CORREZIONE: Inizializzazione uniforme standard (C++17)
        { {0.0, 0.0, -4.0}, 0.0 }, // 1. Decollo (4m)
        { {10.0, 0.0, -4.0}, M_PI/2.0 }, // 2. Waypoint 1
        { {10.0, 10.0, -4.0}, M_PI }, // 3. Waypoint 2
        { {0.0, 10.0, -6.0}, -M_PI/2.0 }, // 4. Waypoint 3 (Salita)
        { {-10.0, 10.0, -6.0}, 0.0 }, // 5. Waypoint 4
        { {-10.0, -10.0, -4.0}, M_PI }, // 6. Waypoint 5 (Discesa)
        { {0.0, 0.0, -4.0}, 0.0 }, // 7. Waypoint 6
        { {0.0, 0.0, -2.0}, 0.0 }, // 8. Land (Hover a 2m)
    };

    // --- CALLBACKS ---
    void vehicle_local_position_callback(const VehicleLocalPosition::SharedPtr msg)
    {
        // Aggiorna posizione locale
        current_position_ << msg->x, msg->y, msg->z;
        // Aggiorna Yaw (semplificato a heading)
        current_yaw_ = msg->heading; 
        // Aggiorna timestamp per il check di connettività
        last_position_timestamp_ = this->now().nanoseconds();
    }

    // --- PUBBLICAZIONI ---
    void publish_vehicle_command(uint32_t command, float param1 = 0.0, float param2 = 0.0)
    {
        VehicleCommand msg{};
        msg.timestamp = this->now().nanoseconds() / 1000;
        msg.command = command;
        msg.param1 = param1;
        msg.param2 = param2;
        msg.target_system = 1;
        msg.target_component = 1;
        msg.source_system = 1;
        msg.source_component = 1;
        msg.from_external = true;
        vehicle_command_publisher_->publish(msg);
    }

    void publish_offboard_control_mode()
    {
        OffboardControlMode msg{};
        msg.timestamp = this->now().nanoseconds() / 1000;
        msg.position = true;
        msg.velocity = false;
        msg.acceleration = false;
        msg.attitude = false;
        msg.body_rate = false;
        offboard_control_mode_publisher_->publish(msg);
    }

    void publish_trajectory_setpoint(const Vector3d& pos, double yaw)
    {
        TrajectorySetpoint msg{};
        msg.timestamp = this->now().nanoseconds() / 1000;
        
        msg.position[0] = (float)pos(0);
        msg.position[1] = (float)pos(1);
        msg.position[2] = (float)pos(2);
        msg.yaw = (float)yaw;

        // Invia setpoints di velocità a NaN per lasciare che PX4 usi la sua velocità di crociera
        msg.velocity[0] = NAN;
        msg.velocity[1] = NAN;
        msg.velocity[2] = NAN;
        
        trajectory_setpoint_publisher_->publish(msg);
    }
    
    double distance_to_waypoint(const Vector3d& target_pos) {
        return (target_pos - current_position_).norm();
    }

    // --- LOOP PRINCIPALE (La Macchina a Stati) ---
    void run_loop()
    {
        // 1. Manteniamo il flusso continuo per il failsafe Offboard
        publish_offboard_control_mode();

        if (is_landing_commanded_ || current_waypoint_index_ >= waypoints_.size()) {
            flight_state_ = FlightState::DONE;
            return;
        }

        Vector3d target_pos = waypoints_[current_waypoint_index_].position;
        double target_yaw = waypoints_[current_waypoint_index_].yaw;

        switch (flight_state_) {
            case FlightState::SETUP:
            {
                // Invia setpoint statico iniziale
                publish_trajectory_setpoint(target_pos, target_yaw);
                
                // Check di connettività
                double time_since_last_pos = (double)(this->now().nanoseconds() - last_position_timestamp_) / 1e9;

                if (offboard_setpoint_counter_ >= 100) { // 5 secondi di setpoints
                    if (last_position_timestamp_ > 0 && time_since_last_pos < 1.0) {
                        // Connessione verificata. Passa alla fase ARMING
                        flight_state_ = FlightState::ARMING;
                        RCLCPP_INFO(this->get_logger(), "Setup completo. Pronti per ARM/OFFBOARD loop.");
                    } else {
                        RCLCPP_WARN(this->get_logger(), "Attendendo connessione PX4/Agent. Ultimo messaggio ricevuto %.2f secondi fa. Rimanere in SETUP.", time_since_last_pos);
                        offboard_setpoint_counter_ = 99; // Rimani in setup finché non connesso
                    }
                }
                break;
            }

            case FlightState::ARMING:
            {
                // Stato B: Invia ARM e OFFBOARD MODE continuamente (ridondanza C++ per la stabilità)
                
                // Azioni critiche ridondanti (20 Hz)
                publish_vehicle_command(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0); // ARM
                publish_vehicle_command(VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0); // OFFBOARD MODE

                // Continua a inviare il setpoint di decollo
                publish_trajectory_setpoint(target_pos, target_yaw);

                // Controllo del Decollo: verifica che la posizione Z sia vicina al target (-4.0m)
                // Se l'armamento ha successo, il drone dovrebbe iniziare a salire.
                // Controlliamo se ha raggiunto almeno 0.5m di altezza per confermare l'armamento.
                if (std::abs(current_position_(2)) > 0.5) { // Altezza > 0.5m (Down è -Z)
                    // Controlla se ha raggiunto l'altezza target
                    if (std::abs(current_position_(2) - target_pos(2)) < 0.5) { 
                        RCLCPP_INFO(this->get_logger(), "Decollo completato. Inizio traiettoria.");
                        current_waypoint_index_ = 1;
                        flight_state_ = FlightState::TRAJECTORY;
                    } else {
                         RCLCPP_INFO(this->get_logger(), "Salendo a %.1f m. Posizione Z corrente: %.1f m",
                                 -target_pos(2), -current_position_(2));
                    }
                }
                break;
            }

            case FlightState::TRAJECTORY:
            {
                // Stato D: Navigazione continua

                // Invia il setpoint corrente (Posizione e Yaw)
                publish_trajectory_setpoint(target_pos, target_yaw);

                // Criterio di transizione: Se siamo vicini al waypoint corrente
                if (distance_to_waypoint(target_pos) < 1.0) { // Tolleranza di 1.0 metro
                    RCLCPP_INFO(this->get_logger(), "Waypoint %zu raggiunto. Prossimo.", current_waypoint_index_);
                    current_waypoint_index_++;
                    
                    if (current_waypoint_index_ == waypoints_.size() - 1) {
                         flight_state_ = FlightState::LANDING_PREP;
                         RCLCPP_INFO(this->get_logger(), "Ultimo Waypoint raggiunto. Preparazione per Landing.");
                    }
                }
                break;
            }

            case FlightState::LANDING_PREP:
            {
                // Stato E: Atterraggio Autonomo (più sicuro del disarmo manuale)
                publish_vehicle_command(VehicleCommand::VEHICLE_CMD_NAV_LAND);
                is_landing_commanded_ = true;
                RCLCPP_INFO(this->get_logger(), "Comando LAND inviato. PX4 gestirà l'atterraggio e il disarmo.");
                flight_state_ = FlightState::DONE;
                break;
            }

            case FlightState::DONE:
            default:
                break;
        }

        offboard_setpoint_counter_++;
    }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OffboardWaypointController>());
    rclcpp::shutdown();
    return 0;
}
