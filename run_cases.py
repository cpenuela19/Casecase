from seneca_routing_model import Case1Base, Case2FiveClientsPerVehicle, Case3BigDistancesSmallDemands, Case4CapacitatedDepots

# Ejecutar Caso 1
case1 = Case1Base(
    clients_file="case_1_base/Clients.csv",
    depots_file="case_1_base/multi_vehicles.csv",
    drone_file="case_1_base/drone_only.csv",
    ev_file="case_1_base/ev_only.csv",
    gas_file="case_1_base/gas_car_only.csv"
)
case1.solve()
case1.save_results("case_1_base_results.csv")

# Ejecutar Caso 2
case2 = Case2FiveClientsPerVehicle(
    clients_file="case_2_5_clients_per_vehicle/Clients.csv",
    depots_file="case_2_5_clients_per_vehicle/Depots.csv",
    vehicles_file="case_2_5_clients_per_vehicle/Vehicles.csv"
)
case2.solve()
case2.save_results("case_2_5_clients_per_vehicle_results.csv")

# Ejecutar Caso 3
case3 = Case3BigDistancesSmallDemands(
    clients_file="case_3_big_distances_small_demands/Clients.csv",
    depots_file="case_3_big_distances_small_demands/Depots.csv",
    vehicles_file="case_3_big_distances_small_demands/Vehicles.csv"
)
case3.solve()
case3.save_results("case_3_big_distances_small_demands_results.csv")

# Ejecutar Caso 4
case4 = Case4CapacitatedDepots(
    clients_file="case_4_capacitated_depots/Clients.csv",
    depots_file="case_4_capacitated_depots/Depots.csv",
    vehicles_file="case_4_capacitated_depots/Vehicles.csv",
    depot_capacities_file="case_4_capacitated_depots/DepotCapacities.csv"
)
case4.solve()
case4.save_results("case_4_capacitated_depots_results.csv")
