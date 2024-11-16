from pyomo.environ import *
import pandas as pd
import numpy as np

class VehicleRoutingModel:
    def __init__(self, clients_file, depots_file, vehicles_file):
        self.clients = pd.read_csv(clients_file)
        self.depots = pd.read_csv(depots_file)
        self.vehicles = pd.read_csv(vehicles_file)

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
                distances[(c, d)] = np.random.normal(loc=5, scale=1)  # Default 5 km promedio
        return distances

    def _objective_rule(self, model):
        return sum(
            model.x[c, d, v] * model.Distance[c, d] * model.CostPerKM[v]
            for c in model.Clients for d in model.Depots for v in model.Vehicles
        )

    def _add_base_constraints(self):
        model = self.model
        # Cada cliente debe ser atendido exactamente por 1 vehículo
        for c in model.Clients:
            model.constraints.add(
                sum(model.x[c, d, v] for d in model.Depots for v in model.Vehicles) == 1
            )
        # Respetar la capacidad de cada vehículo
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
        # Cargar datos
        self.clients = pd.read_csv(clients_file)
        self.depots = pd.read_csv(depots_file)
        self.drone_vehicles = pd.read_csv(drone_file)
        self.ev_vehicles = pd.read_csv(ev_file)
        self.gas_vehicles = pd.read_csv(gas_file)

        # Combinar vehículos
        self.vehicles = pd.concat([self.drone_vehicles, self.ev_vehicles, self.gas_vehicles], ignore_index=True)

        # Ajustar demandas y distancias
        self.clients['Demand'] = np.random.randint(8, 21, size=len(self.clients))
        self.clients['Distance'] = np.random.normal(loc=5, scale=1, size=len(self.clients))

        self.model = ConcreteModel()
        self._initialize_sets()
        self._initialize_parameters()
        self._initialize_variables()
        self._define_objective()
        self._define_constraints()

    def _add_base_constraints(self):
        super()._add_base_constraints()
        model = self.model
        # Cada vehículo puede atender máximo 2 clientes
        for v in model.Vehicles:
            model.constraints.add(
                sum(model.x[c, d, v] for c in model.Clients for d in model.Depots) <= 2
            )

class Case2FiveClientsPerVehicle(VehicleRoutingModel):
    def _add_base_constraints(self):
        super()._add_base_constraints()
        model = self.model
        # Cada vehículo puede atender máximo 5 clientes
        for v in model.Vehicles:
            model.constraints.add(
                sum(model.x[c, d, v] for c in model.Clients for d in model.Depots) <= 5
            )

class Case3BigDistancesSmallDemands(VehicleRoutingModel):
    def __init__(self, clients_file, depots_file, vehicles_file):
        super().__init__(clients_file, depots_file, vehicles_file)

        # Ajustar distancias y demandas
        self.clients['Distance'] = np.random.normal(loc=10, scale=0.5, size=len(self.clients))
        self.clients['Demand'] = np.random.randint(1, 6, size=len(self.clients))

    def _initialize_parameters(self):
        super()._initialize_parameters()
        self.model.Distance = Param(
            self.model.Clients,
            self.model.Depots,
            initialize=self._calculate_distances()
        )

    def _calculate_distances(self):
        distances = {}
        for c in self.clients['ClientID']:
            for d in self.depots['DepotID']:
                distances[(c, d)] = np.random.normal(loc=10, scale=0.5)  # Promedio 10 km
        return distances

class Case4CapacitatedDepots(VehicleRoutingModel):
    def __init__(self, clients_file, depots_file, vehicles_file, depot_capacities_file):
        super().__init__(clients_file, depots_file, vehicles_file)

        # Cargar capacidades de los centros
        self.depot_capacities = pd.read_csv(depot_capacities_file)
        self.model.DepotCapacity = Param(
            self.model.Depots,
            initialize=self._load_param(self.depot_capacities, 'DepotID', 'Capacity')
        )

class Case4CapacitatedDepots(VehicleRoutingModel):
    def __init__(self, clients_file, depots_file, vehicles_file, depot_capacities_file):
        super().__init__(clients_file, depots_file, vehicles_file)

        # Cargar capacidades de los centros
        self.depot_capacities = pd.read_csv(depot_capacities_file)
        self.model.DepotCapacity = Param(
            self.model.Depots,
            initialize=self._load_param(self.depot_capacities, 'DepotID', 'Capacity')
        )

    def _add_base_constraints(self):
        super()._add_base_constraints()
        model = self.model

        # Restringir capacidad máxima de cada centro
        for d in model.Depots:
            model.constraints.add(
                sum(
                    model.x[c, d, v] * model.Demand[c] for c in model.Clients for v in model.Vehicles
                ) <= model.DepotCapacity[d]
            )
