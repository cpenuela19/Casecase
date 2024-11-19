from pyomo.environ import *
from pyomo.opt import SolverFactory
import pandas as pd
import numpy as np

class VehicleRoutingModel:
    def __init__(self, clients_file, depots_file, vehicles_file, column_mapping):
        """
        Modelo base de ruteo con soporte para mapeo dinámico de columnas.
        """
        self.column_mapping = column_mapping

        # Cargar y mapear archivos
        self.clients = pd.read_csv(clients_file, encoding='utf-8')
        self.clients.columns = self.clients.columns.str.strip()
        self.clients.rename(columns=self.column_mapping.get('clients', {}), inplace=True)

        self.depots = pd.read_csv(depots_file, encoding='utf-8')
        self.depots.columns = self.depots.columns.str.strip()
        self.depots.rename(columns=self.column_mapping.get('depots', {}), inplace=True)

        self.vehicles = pd.read_csv(vehicles_file, encoding='utf-8')
        self.vehicles.columns = self.vehicles.columns.str.strip()
        self.vehicles.rename(columns=self.column_mapping.get('vehicles', {}), inplace=True)

        # Agregar CostPerKM si no está presente
        if 'CostPerKM' not in self.vehicles.columns:
            vehicle_costs = {
                'Gas Car': 0.7,
                'Electric Car': 0.5,
                'Drone': 0.3
            }
            self.vehicles['CostPerKM'] = self.vehicles['VehicleID'].map(vehicle_costs)
            self.vehicles['CostPerKM'].fillna(0.6, inplace=True)  # Valor por defecto si no se encuentra en el diccionario

        # Diagnósticos
        print("Loaded clients:", self.clients.head())
        print("Clients DataFrame Columns:", self.clients.columns.tolist())
        print("Loaded depots:", self.depots.head())
        print("Depots DataFrame Columns:", self.depots.columns.tolist())
        print("Loaded vehicles:", self.vehicles.head())
        print("Vehicles DataFrame Columns:", self.vehicles.columns.tolist())

        # Verificar columnas necesarias
        required_client_cols = ['ClientID', 'Demand', 'Longitude', 'Latitude']
        required_depot_cols = ['DepotID', 'Longitude', 'Latitude']
        required_vehicle_cols = ['VehicleID', 'Capacity', 'CostPerKM']
        if not set(required_client_cols).issubset(self.clients.columns):
            raise ValueError(f"Clients file must contain columns: {required_client_cols}")
        if not set(required_depot_cols).issubset(self.depots.columns):
            raise ValueError(f"Depots file must contain columns: {required_depot_cols}")
        if not set(required_vehicle_cols).issubset(self.vehicles.columns):
            raise ValueError(f"Vehicles file must contain columns: {required_vehicle_cols}")

        # Inicializar modelo
        self.model = ConcreteModel()
        self._initialize_sets()
        self._initialize_parameters()
        self._initialize_variables()
        self._define_objective()
        self._define_constraints()

    def _initialize_sets(self):
        """
        Inicializar conjuntos.
        """
        self.model.Clients = Set(initialize=self.clients['ClientID'].tolist())
        self.model.Depots = Set(initialize=self.depots['DepotID'].tolist())
        self.model.Vehicles = Set(initialize=self.vehicles['VehicleID'].tolist())

    def _initialize_parameters(self):
        """
        Inicializar parámetros.
        """
        self.model.Demand = Param(self.model.Clients, initialize=self._load_param(self.clients, 'ClientID', 'Demand'))
        self.model.Capacity = Param(self.model.Vehicles, initialize=self._load_param(self.vehicles, 'VehicleID', 'Capacity'))
        self.model.CostPerKM = Param(self.model.Vehicles, initialize=self._load_param(self.vehicles, 'VehicleID', 'CostPerKM'))
        self.model.Distance = Param(
            self.model.Clients,
            self.model.Depots,
            initialize=self._calculate_distances()
        )

    def _initialize_variables(self):
        """
        Inicializar variables de decisión.
        """
        self.model.x = Var(self.model.Clients, self.model.Depots, self.model.Vehicles, within=Binary)

    def _define_objective(self):
        """
        Definir objetivo.
        """
        self.model.obj = Objective(rule=self._objective_rule, sense=minimize)

    def _define_constraints(self):
        """
        Definir restricciones.
        """
        self.model.constraints = ConstraintList()
        self._add_base_constraints()

    def _load_param(self, dataframe, index_col, value_col):
        """
        Convertir columnas de DataFrame en un diccionario de parámetros Pyomo.
        """
        return dataframe.set_index(index_col)[value_col].to_dict()

    def _calculate_distances(self):
        distances = {}
        for c in self.model.Clients:
            # Obtener coordenadas del cliente
            client_row = self.clients[self.clients['ClientID'] == c].iloc[0]
            client_coords = (client_row['Latitude'], client_row['Longitude'])
            for d in self.model.Depots:
                # Obtener coordenadas del depósito
                depot_row = self.depots[self.depots['DepotID'] == d].iloc[0]
                depot_coords = (depot_row['Latitude'], depot_row['Longitude'])
                # Calcular distancia euclidiana
                distance = np.sqrt(
                    (client_coords[0] - depot_coords[0])**2 +
                    (client_coords[1] - depot_coords[1])**2
                )
                distances[(c, d)] = distance
        return distances

    def _objective_rule(self, model):
        """
        Función objetivo para minimizar el costo total.
        """
        return sum(
            model.x[c, d, v] * model.Distance[c, d] * model.CostPerKM[v]
            for c in model.Clients for d in model.Depots for v in model.Vehicles
        )

    def _add_base_constraints(self):
        """
        Añadir restricciones básicas.
        """
        model = self.model
        # Cada cliente debe ser servido por exactamente un vehículo y un depósito
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
        """
        Resolver el problema.
        """
        opt = SolverFactory(solver)
        results = opt.solve(self.model, tee=True)

        # Validar si la solución es factible
        if results.solver.termination_condition != TerminationCondition.optimal:
            print("\nWARNING: No feasible solution found. Check the constraints and inputs.")
            return results

        print("\nSolution found successfully.")
        return results

    def save_results(self, output_file):
        """
        Guardar resultados en un archivo CSV.
        """
        results = []
        for c in self.model.Clients:
            for d in self.model.Depots:
                for v in self.model.Vehicles:
                    val = self.model.x[c, d, v]()
                    if val is not None and val > 0.5:
                        results.append({'Client': c, 'Depot': d, 'Vehicle': v})
        if not results:
            print("No valid assignments found. Check model constraints and inputs.")
        else:
            pd.DataFrame(results).to_csv(output_file, index=False)

class Case2FiveClientsPerVehicle(VehicleRoutingModel):
    def _initialize_sets(self):
        super()._initialize_sets()

    def _initialize_parameters(self):
        super()._initialize_parameters()

    def _add_base_constraints(self):
        super()._add_base_constraints()
        model = self.model
        # Cada vehículo puede servir un máximo de 5 clientes
        for v in model.Vehicles:
            model.constraints.add(
                sum(model.x[c, d, v] for c in model.Clients for d in model.Depots) <= 5
            )


case_2_mapping = {
    'clients': {'Product': 'Demand'},
    'depots': {}, 
    'vehicles': {'VehicleType': 'VehicleID'}  
}

# Execute Case 2
case2 = Case2FiveClientsPerVehicle(
    clients_file=r"C:\Users\cpenu\Documents\6-semestre\MoS\Proyecto2-MoS\case_2_5_clients_per_vehicle\case_2_5_clients_per_vehicle\Clients.csv",
    depots_file=r"C:\Users\cpenu\Documents\6-semestre\MoS\Proyecto2-MoS\case_2_5_clients_per_vehicle\case_2_5_clients_per_vehicle\Depots.csv",
    vehicles_file=r"C:\Users\cpenu\Documents\6-semestre\MoS\Proyecto2-MoS\case_2_5_clients_per_vehicle\case_2_5_clients_per_vehicle\Vehicles.csv",
    column_mapping=case_2_mapping
)
case2.solve()
case2.save_results("case_2_5_clients_per_vehicle_results.csv")
