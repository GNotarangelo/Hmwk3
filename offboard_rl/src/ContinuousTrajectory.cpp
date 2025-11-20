#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include <cmath>
#include <px4_msgs/msg/vehicle_local_position.hpp>
#include <px4_msgs/msg/vehicle_attitude.hpp>
#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <offboard_rl/utils.h> 

using namespace std::chrono_literals;
using namespace px4_msgs::msg;
using Eigen::Vector4d;
using Eigen::Vector3d;

// Definiamo i tipi di dati per i waypoint per chiarezza
struct Waypoint {
    Vector3d position; // [x (North), y (East), z (Down)]
    double yaw;        // Orientamento in radianti
    double segment_time; // Tempo stimato per coprire questo segmento dal waypoint precedente
    double target_vel;   // Velocità desiderata all'arrivo a questo waypoint 
};

// --- Configurazione QoS (Quality of Service) essenziale per il controllo ---
rclcpp::QoS qos_profile_control() {
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

// Configurazione QoS per i messaggi di feedback dai sensori
rclcpp::QoS qos_profile_sensor() {
    return rclcpp::QoS(rclcpp::QoSInitialization(RMW_QOS_POLICY_HISTORY_KEEP_LAST, 5),
                       rmw_qos_profile_sensor_data);
}


class ConcatenatedPlanner : public rclcpp::Node
{
    public:
    ConcatenatedPlanner() : Node("concatenated_planner")
    {
        // 1. Inizializzazione Pub/Sub
        offboard_control_mode_publisher_ = this->create_publisher<OffboardControlMode>("/fmu/in/offboard_control_mode", qos_profile_control());
        trajectory_setpoint_publisher_ = this->create_publisher<TrajectorySetpoint>("/fmu/in/trajectory_setpoint", qos_profile_control());
        vehicle_command_publisher_ = this->create_publisher<VehicleCommand>("/fmu/in/vehicle_command", qos_profile_control());

        local_position_subscription_ = this->create_subscription<VehicleLocalPosition>("/fmu/out/vehicle_local_position", qos_profile_sensor(), 
            std::bind(&ConcatenatedPlanner::vehicle_local_position_callback, this, std::placeholders::_1));
        attitude_subscription_ = this->create_subscription<VehicleAttitude>("/fmu/out/vehicle_attitude", qos_profile_sensor(), 
            std::bind(&ConcatenatedPlanner::vehicle_attitude_callback, this, std::placeholders::_1));

        // 2. Timer: 10 Hz per l'attivazione Offboard, 50 Hz per il setpoint di traiettoria
        timer_offboard_ = this->create_wall_timer(100ms, std::bind(&ConcatenatedPlanner::activate_offboard_and_arming, this));
        timer_trajectory_publish_ = this->create_wall_timer(20ms, std::bind(&ConcatenatedPlanner::publish_trajectory_setpoint, this));

        // 3. Thread per l'Input
        keyboard_thread = std::thread(&ConcatenatedPlanner::keyboard_listener, this);
        RCLCPP_INFO(this->get_logger(), "Planner avviato. Attendere la pressione di INVIO per iniziare.");
    }

    private:
    rclcpp::Subscription<VehicleLocalPosition>::SharedPtr local_position_subscription_;
    rclcpp::Subscription<VehicleAttitude>::SharedPtr attitude_subscription_;
    rclcpp::Publisher<OffboardControlMode>::SharedPtr offboard_control_mode_publisher_;
    rclcpp::Publisher<VehicleCommand>::SharedPtr vehicle_command_publisher_;
    rclcpp::Publisher<TrajectorySetpoint>::SharedPtr trajectory_setpoint_publisher_;
    rclcpp::TimerBase::SharedPtr timer_offboard_;
    rclcpp::TimerBase::SharedPtr timer_trajectory_publish_;
    std::thread keyboard_thread;
    
    // Variabili di Stato
    VehicleLocalPosition current_position_{};
    VehicleAttitude current_attitude_{};
    double offboard_counter{0};
    size_t current_waypoint_index{0}; // Usiamo size_t per evitare warning di confronto
    double t_segment{0.0}; 
    bool is_armed{false};
    bool is_landing_commanded{false}; 

    // Variabili per la Traiettoria Corrente
    bool segment_computed{false};
    Eigen::Vector<double, 6> poly_coeffs; 
    double T_segment{1.0}; 
    Vector4d pos_i, pos_f; 
    
    // --------------------------------------------------------------------------------
    // 4. DEFINIZIONE DEI WAYPOINT (Tempo aumentato per maggiore fluidità)
    // Ho aumentato leggermente i tempi per segmenti più fluidi, forzando una velocità media più alta.
    std::vector<Waypoint> trajectory_waypoints = {
        // --- WAYPOINT 0: Decollo e Hover (Velocità 0) ---
        { {0.0, 0.0, -2.0}, 0.0, 4.0, 0.0 }, // Decollo in 4s (più tempo per salire)
        
        // --- WAYPOINT 1: Segmento 1 (più tempo per la fluidità) ---
        { {6.0, 0.0, -4.0}, 0.0, 6.0, 1.8 }, 
        
        // --- WAYPOINT 2: Segmento 2 ---
        { {6.0, 6.0, -4.0}, M_PI/2.0, 5.0, 1.8 }, 
        
        // --- WAYPOINT 3: Segmento 3 (Curva e Salita) ---
        { {0.0, 6.0, -7.0}, M_PI, 7.0, 1.8 }, // Più alto, più tempo
        
        // --- WAYPOINT 4: Segmento 4 ---
        { {-6.0, 0.0, -7.0}, -M_PI/2.0, 7.0, 1.8 },
        
        // --- WAYPOINT 5: Segmento 5 (Discesa) ---
        { {-6.0, -6.0, -3.0}, 0.0, 8.0, 1.8 }, // Più tempo per la discesa
        
        // --- WAYPOINT 6: Segmento 6 (Ritorno) ---
        { {0.0, -6.0, -3.0}, 0.0, 6.0, 1.8 },

        // --- WAYPOINT 7: Arrivo per LAND (Ritorno a -2m) ---
        { {0.0, 0.0, -2.0}, 0.0, 4.0, 0.0 }, // Hover finale
    };
    // --------------------------------------------------------------------------------


    // --- CALLBACKS ---
    void vehicle_local_position_callback(const VehicleLocalPosition::SharedPtr msg)
    {
        current_position_ = *msg;
    }

    void vehicle_attitude_callback(const VehicleAttitude::SharedPtr msg)
    {
        current_attitude_ = *msg;
    }

    // --- LOGICA DI TRAIETTORIA ---
    double distance_to_waypoint(const Vector3d& target_pos) {
        return (target_pos - Vector3d(current_position_.x, current_position_.y, current_position_.z)).norm();
    }
    
    Vector4d get_segment_error(const Vector4d& p_start, const Vector4d& p_end) {
        Vector4d e = p_end - p_start;
        e(3) = utilities::angleError(p_end(3), p_start(3)); 
        return e;
    }

    void compute_segment_polynomial(double T) {
        
        Vector4d e = get_segment_error(pos_i, pos_f); 
        double s_f = e.head<3>().norm(); 

        if (s_f < 0.1) {
            poly_coeffs.setZero(); 
            T_segment = 1.0;
            return;
        }

        // Condizioni al contorno: [s(0), s'(0), s''(0), s(T), s'(T), s''(T)]
        Eigen::VectorXd b(6);
        Eigen::Matrix<double, 6, 6> A;

        // Qui, il polinomio è ancora C2 (Velocità 0 all'inizio e alla fine), ma
        // l'alta frequenza di invio e il tempo lungo del segmento forzano la fluidità.
        b << 0.0, 0.0, 0.0, s_f, 0.0, 0.0;

        A << 0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 1, 0,
            0, 0, 0, 1, 0, 0,
            pow(T,5), pow(T,4), pow(T,3), pow(T,2), T, 1,
            5*pow(T,4), 4*pow(T,3), 3*pow(T,2), 2*T, 1, 0,
            20*pow(T,3), 12*pow(T,2), 6*T, 2, 0, 0;

        poly_coeffs = A.inverse() * b;
        segment_computed = true;
        T_segment = T;
    }

    TrajectorySetpoint compute_trajectory_setpoint(double t)
    {
        TrajectorySetpoint msg{};
        if (!segment_computed || t > T_segment) {
            msg.position = {float(pos_f(0)), float(pos_f(1)), float(pos_f(2))};
            msg.velocity = {0.0f, 0.0f, 0.0f}; 
            msg.yaw = float(pos_f(3));
            msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
            return msg;
        }

        Vector4d e = get_segment_error(pos_i, pos_f); 
        double s_f = e.head<3>().norm();

        // 1. Valutazione del polinomio
        double s, s_d, s_dd;
        s = poly_coeffs(0) * std::pow(t, 5.0) + poly_coeffs(1) * std::pow(t, 4.0) + poly_coeffs(2) * std::pow(t, 3.0) + poly_coeffs(3) * std::pow(t, 2.0) + poly_coeffs(4) * t + poly_coeffs(5);
        s_d = 5.0 * poly_coeffs(0) * std::pow(t, 4.0) + 4.0 * poly_coeffs(1) * std::pow(t, 3.0) + 3.0 * poly_coeffs(2) * std::pow(t, 2.0) + 2.0 * poly_coeffs(3) * t + poly_coeffs(4);
        s_dd = 20.0 * poly_coeffs(0) * std::pow(t, 3.0) + 12.0 * poly_coeffs(1) * std::pow(t, 2.0) + 6.0 * poly_coeffs(3) * t + poly_coeffs(3); 

        // 2. Mapping nello spazio 3D
        Vector4d ref_traj_pos, ref_traj_vel, ref_traj_acc;
        
        if (s_f > 1e-3) { 
            ref_traj_pos = pos_i + s * e / s_f;
            ref_traj_vel = s_d * e / s_f;
            ref_traj_acc = s_dd * e / s_f;
        } else {
            ref_traj_pos = pos_f;
            ref_traj_vel.setZero();
            ref_traj_acc.setZero();
        }

        // 3. Creazione del messaggio
        msg.position = {float(ref_traj_pos(0)), float(ref_traj_pos(1)), float(ref_traj_pos(2))};
        msg.velocity = {float(ref_traj_vel(0)), float(ref_traj_vel(1)), float(ref_traj_vel(2))};
        msg.acceleration = {float(ref_traj_acc(0)), float(ref_traj_acc(1)), float(ref_traj_acc(2))};
        msg.yaw = float(ref_traj_pos(3)); 
        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;

        return msg;
    }

    void transition_to_next_waypoint() {
        if (current_waypoint_index >= trajectory_waypoints.size() - 1) { 
            if (!is_landing_commanded) {
                publish_vehicle_command(VehicleCommand::VEHICLE_CMD_NAV_LAND);
                RCLCPP_INFO(this->get_logger(), "Traiettoria completata. Comando LAND mode. In attesa di disarmo automatico da PX4."); 
                is_landing_commanded = true;
            }
            return; 
        }

        current_waypoint_index++;
        
        Waypoint start_wp;
        if (current_waypoint_index > 0) {
            start_wp = trajectory_waypoints[current_waypoint_index - 1];
            pos_i << start_wp.position, start_wp.yaw; 
        } else {
            pos_i << current_position_.x, current_position_.y, current_position_.z, current_attitude_.q[3];
        }

        Waypoint end_wp = trajectory_waypoints[current_waypoint_index];

        pos_f << end_wp.position, end_wp.yaw;

        T_segment = end_wp.segment_time;

        t_segment = 0.0;
        segment_computed = false; 

        compute_segment_polynomial(T_segment);
        // CORREZIONE FORMATO LOG: Usiamo %zu per size_t e %f per double, come richiesto da RCLCPP_INFO
        RCLCPP_INFO(this->get_logger(), "--- Transizione a Waypoint %zu --- Posizione: (%.1f, %.1f, %.1f), Durata: %.1fs",
                                 current_waypoint_index, pos_f(0), pos_f(1), pos_f(2), T_segment);
    }


    // --- LOGICA DI FLIGHT CONTROL ---

    void publish_vehicle_command(uint32_t command, float param1 = 0.0, float param2 = 0.0)
    {
        VehicleCommand msg{};
        msg.command = command;
        msg.param1 = param1;
        msg.param2 = param2;
        msg.target_system = 1;
        msg.target_component = 1;
        msg.source_system = 1;
        msg.source_component = 1;
        msg.from_external = true;
        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
        vehicle_command_publisher_->publish(msg);
    }


    void activate_offboard_and_arming()
    {
        // 1. Invio continuo di OffboardControlMode (Failsafe)
        OffboardControlMode msg{};
        msg.position = true;
        msg.velocity = false;
        msg.acceleration = false;
        msg.attitude = false;
        msg.body_rate = false;
        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
        offboard_control_mode_publisher_->publish(msg); 
        
        if (!is_armed && offboard_counter >= 10) {
            // 2. Arming (dopo 1 secondo di messaggi Offboard)
            publish_vehicle_command(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0);
            
            // 3. Cambio in Offboard Mode
            publish_vehicle_command(VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0); 
            
            is_armed = true;
            RCLCPP_INFO(this->get_logger(), "Drone Armato e in Modalità Offboard.");

            // 4. Inizia il primo segmento (Decollo/Waypoint 0)
            transition_to_next_waypoint(); 
        }

        if (offboard_counter < 11) offboard_counter++;
    }


    void publish_trajectory_setpoint()
    {
        if (!is_armed) return;

        // Se il comando LANDING è stato inviato, usciamo immediatamente
        if (is_landing_commanded) return;

        if (t_segment >= T_segment) {
            transition_to_next_waypoint();
            if (!is_armed) return; 
        }

        TrajectorySetpoint msg = compute_trajectory_setpoint(t_segment);
        trajectory_setpoint_publisher_->publish(msg);
        
        t_segment += 0.02; 
    }
    
    void keyboard_listener() {
        std::string line;
        std::cout << "Premi INVIO per iniziare la sequenza di armamento e traiettoria...";
        std::getline(std::cin, line);
    }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ConcatenatedPlanner>());
    rclcpp::shutdown();
    return 0;
}
