from pyomo.environ import *
import pandas as pd
import numpy as np

class VehicleRoutingModel:
    def __init__(self, clients_file, depots_file, vehicles_file):
        # Load files
        self.clients = pd.read_csv(clients_file, encoding='utf-8')
        self.clients.columns = self.clients.columns.str.strip()

        self.depots = pd.read_csv(depots_file, encoding='utf-8')
        self.depots.columns = self.depots.columns.str.strip()

        self.vehicles = pd.read_csv(vehicles_file, encoding='utf-8')
        self.vehicles.columns = self.vehicles.columns.str.strip()

        # Validate headers
        required_columns_depots = {'DepotID', 'Longitude', 'Latitude'}
        if not required_columns_depots.issubset(self.depots.columns):
            raise ValueError(f"The Depots.csv file must contain the columns: {required_columns_depots}. "
                             f"Found columns: {', '.join(self.depots.columns)}")

        # Diagnostics
        print("Loaded clients:", self.clients.head())
        print("Loaded depots:", self.depots.head())
        print("Loaded vehicles:", self.vehicles.head())

        # Initialize the model
        self.model = ConcreteModel()
        self._initialize_sets()
        self._initialize_parameters()
        self._initialize_variables()
        self._define_objective()
        self._define_constraints()

    def _initialize_sets(self):
        self.model.Clients = Set(initialize=self.clients['ClientID'].tolist())
        self.model.Depots = Set(initialize=self.depots['DepotID'].tolist())
        self.model.Vehicles = Set(initialize=self.vehicles['VehicleID'].tolist())

    def _initialize_parameters(self):
        self.model.Demand = Param(self.model.Clients, initialize=self._load_param(self.clients, 'ClientID', 'Demand'))
        self.model.Capacity = Param(self.model.Vehicles, initialize=self._load_param(self.vehicles, 'VehicleID', 'Capacity'))
        self.model.CostPerKM = Param(self.model.Vehicles, initialize=self._load_param(self.vehicles, 'VehicleID', 'FreightRate'))
        self.model.Distance = Param(
            self.model.Clients,
            self.model.Depots,
            initialize=self._calculate_distances()
        )

    def _initialize_variables(self):
        self.model.x = Var(self.model.Clients, self.model.Depots, self.model.Vehicles, within=Binary)

    def _define_objective(self):
        self.model.obj = Objective(rule=self._objective_rule, sense=minimize)

    def _define_constraints(self):
        self.model.constraints = ConstraintList()
        self._add_base_constraints()

    def _load_param(self, dataframe, index_col, value_col):
        return dataframe.set_index(index_col)[value_col].to_dict()

    def _calculate_distances(self):
        distances = {}
        for c in self.clients['ClientID']:
            for d in self.depots['DepotID']:
                distances[(c, d)] = np.random.normal(loc=5, scale=1)  # Default average of 5 km
        return distances

    def _objective_rule(self, model):
        return sum(
            model.x[c, d, v] * model.Distance[c, d] * model.CostPerKM[v]
            for c in model.Clients for d in model.Depots for v in model.Vehicles
        )

    def _add_base_constraints(self):
        model = self.model
        # Each client must be served by exactly one vehicle
        for c in model.Clients:
            model.constraints.add(
                sum(model.x[c, d, v] for d in model.Depots for v in model.Vehicles) == 1
            )
        # Respect the capacity of each vehicle
        for v in model.Vehicles:
            model.constraints.add(
                sum(model.x[c, d, v] * model.Demand[c] for c in model.Clients for d in model.Depots) <= model.Capacity[v]
            )

    def solve(self, solver='glpk'):
        opt = SolverFactory(solver)
        return opt.solve(self.model, tee=True)

    def save_results(self, output_file):
        results = []
        for c in self.model.Clients:
            for d in self.model.Depots:
                for v in self.model.Vehicles:
                    if self.model.x[c, d, v]() > 0.5:
                        results.append({'Client': c, 'Depot': d, 'Vehicle': v})
        pd.DataFrame(results).to_csv(output_file, index=False)

class Case1Base(VehicleRoutingModel):
    def __init__(self, clients_file, depots_file, drone_file, ev_file, gas_file):
        # Load data
        self.clients = pd.read_csv(clients_file)
        self.clients.columns = self.clients.columns.str.strip()

        self.depots = pd.read_csv(depots_file)
        self.depots.columns = self.depots.columns.str.strip()

        self.drone_vehicles = pd.read_csv(drone_file)
        self.drone_vehicles.columns = self.drone_vehicles.columns.str.strip()

        self.ev_vehicles = pd.read_csv(ev_file)
        self.ev_vehicles.columns = self.ev_vehicles.columns.str.strip()

        self.gas_vehicles = pd.read_csv(gas_file)
        self.gas_vehicles.columns = self.gas_vehicles.columns.str.strip()

        # Combine vehicles
        self.vehicles = pd.concat([self.drone_vehicles, self.ev_vehicles, self.gas_vehicles], ignore_index=True)

        # Adjust demands and distances
        self.clients['Demand'] = np.random.randint(8, 21, size=len(self.clients))
        self.clients['Distance'] = np.random.normal(loc=5, scale=1, size=len(self.clients))

        # Initialize the model
        self.model = ConcreteModel()
        self._initialize_sets()
        self._initialize_parameters()
        self._initialize_variables()
        self._define_objective()
        self._define_constraints()

    def _initialize_sets(self):
        self.model.Clients = Set(initialize=self.clients['ClientID'].tolist())
        self.model.Depots = Set(initialize=self.depots['DepotID'].tolist())
        self.model.Vehicles = Set(initialize=self.vehicles['VehicleID'].tolist())

    def _initialize_parameters(self):
        self.model.Demand = Param(self.model.Clients, initialize=self._load_param(self.clients, 'ClientID', 'Demand'))
        self.model.Capacity = Param(self.model.Vehicles, initialize=self._load_param(self.vehicles, 'VehicleID', 'Capacity'))
        self.model.CostPerKM = Param(self.model.Vehicles, initialize=self._load_param(self.vehicles, 'VehicleID', 'FreightRate'))
        self.model.Distance = Param(
            self.model.Clients,
            self.model.Depots,
            initialize=self._calculate_distances()
        )

    def _calculate_distances(self):
        distances = {}
        for c in self.clients['ClientID']:
            for d in self.depots['DepotID']:
                distances[(c, d)] = np.random.normal(loc=5, scale=1)  # Average of 5 km
        return distances

    def _add_base_constraints(self):
        super()._add_base_constraints()
        model = self.model
        # Each vehicle can serve a maximum of 2 clients
        for v in model.Vehicles:
            model.constraints.add(
                sum(model.x[c, d, v] for c in model.Clients for d in model.Depots) <= 2
            )

class Case2FiveClientsPerVehicle(VehicleRoutingModel):
    def _add_base_constraints(self):
        super()._add_base_constraints()
        model = self.model
        # Each vehicle can serve a maximum of 5 clients
        for v in model.Vehicles:
            model.constraints.add(
                sum(model.x[c, d, v] for c in model.Clients for d in model.Depots) <= 5
            )

class Case3BigDistancesSmallDemands(VehicleRoutingModel):
    def __init__(self, clients_file, depots_file, vehicles_file):
        super().__init__(clients_file, depots_file, vehicles_file)

        # Adjust distances and demands
        self.clients['Distance'] = np.random.normal(loc=10, scale=0.5, size=len(self.clients))
        self.clients['Demand'] = np.random.randint(1, 6, size=len(self.clients))

        # Re-initialize parameters to reflect updated demands and distances
        self._initialize_parameters()

    def _initialize_parameters(self):
        super()._initialize_parameters()
        # Override the Distance parameter with new distances
        self.model.Distance = Param(
            self.model.Clients,
            self.model.Depots,
            initialize=self._calculate_distances()
        )

    def _calculate_distances(self):
        distances = {}
        for c in self.clients['ClientID']:
            for d in self.depots['DepotID']:
                distances[(c, d)] = np.random.normal(loc=10, scale=0.5)  # Average of 10 km
        return distances

class Case4CapacitatedDepots(VehicleRoutingModel):
    def __init__(self, clients_file, depots_file, vehicles_file, depot_capacities_file):
        super().__init__(clients_file, depots_file, vehicles_file)

        # Load depot capacities
        self.depot_capacities = pd.read_csv(depot_capacities_file)
        self.depot_capacities.columns = self.depot_capacities.columns.str.strip()
        self.model.DepotCapacity = Param(
            self.model.Depots,
            initialize=self._load_param(self.depot_capacities, 'DepotID', 'Capacity')
        )

    def _add_base_constraints(self):
        super()._add_base_constraints()
        model = self.model

        # Restrict maximum capacity of each depot
        for d in model.Depots:
            model.constraints.add(
                sum(
                    model.x[c, d, v] * model.Demand[c] for c in model.Clients for v in model.Vehicles
                ) <= model.DepotCapacity[d]
            )

# Execute Case 1
case1 = Case1Base(
    clients_file=r"C:\Users\cpenu\Documents\6-semestre\MoS\MoS Proyecto2\case_1_base\Clients.csv",
    depots_file=r"C:\Users\cpenu\Documents\6-semestre\MoS\MoS Proyecto2\case_1_base\Depots.csv",
    drone_file=r"C:\Users\cpenu\Documents\6-semestre\MoS\MoS Proyecto2\case_1_base\drone_only.csv",
    ev_file=r"C:\Users\cpenu\Documents\6-semestre\MoS\MoS Proyecto2\case_1_base\ev_only.csv",
    gas_file=r"C:\Users\cpenu\Documents\6-semestre\MoS\MoS Proyecto2\case_1_base\gas_car_only.csv"
)
case1.solve()
case1.save_results("case_1_base_results.csv")

# Execute Case 2
case2 = Case2FiveClientsPerVehicle(
    clients_file=r"C:\Users\cpenu\Documents\6-semestre\MoS\MoS Proyecto2\case_2_5_clients_per_vehicle\Clients.csv",
    depots_file=r"C:\Users\cpenu\Documents\6-semestre\MoS\MoS Proyecto2\case_2_5_clients_per_vehicle\Depots.csv",
    vehicles_file=r"C:\Users\cpenu\Documents\6-semestre\MoS\MoS Proyecto2\case_2_5_clients_per_vehicle\Vehicles.csv"
)
case2.solve()
case2.save_results("case_2_5_clients_per_vehicle_results.csv")

# Execute Case 3
case3 = Case3BigDistancesSmallDemands(
    clients_file=r"C:\Users\cpenu\Documents\6-semestre\MoS\MoS Proyecto2\case_3_big_distances_small_demands\Clients.csv",
    depots_file=r"C:\Users\cpenu\Documents\6-semestre\MoS\MoS Proyecto2\case_3_big_distances_small_demands\Depots.csv",
    vehicles_file=r"C:\Users\cpenu\Documents\6-semestre\MoS\MoS Proyecto2\case_3_big_distances_small_demands\Vehicles.csv"
)
case3.solve()
case3.save_results("case_3_big_distances_small_demands_results.csv")

# Execute Case 4
case4 = Case4CapacitatedDepots(
    clients_file=r"C:\Users\cpenu\Documents\6-semestre\MoS\MoS Proyecto2\case_4_capacitated_depots\Clients.csv",
    depots_file=r"C:\Users\cpenu\Documents\6-semestre\MoS\MoS Proyecto2\case_4_capacitated_depots\Depots.csv",
    vehicles_file=r"C:\Users\cpenu\Documents\6-semestre\MoS\MoS Proyecto2\case_4_capacitated_depots\Vehicles.csv",
    depot_capacities_file=r"C:\Users\cpenu\Documents\6-semestre\MoS\MoS Proyecto2\case_4_capacitated_depots\DepotCapacities.csv"
)
case4.solve()
case4.save_results("case_4_capacitated_depots_results.csv")
