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

        # Diagnósticos
        print("Loaded clients:", self.clients.head())
        print("Clients DataFrame Columns:", self.clients.columns.tolist())
        print("Loaded depots:", self.depots.head())
        print("Depots DataFrame Columns:", self.depots.columns.tolist())
        print("Loaded vehicles:", self.vehicles.head())
        print("Vehicles DataFrame Columns:", self.vehicles.columns.tolist())

        # Verificar columnas necesarias
        required_client_cols = ['ClientID', 'Demand']
        required_depot_cols = ['DepotID']
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
        self.model.CostPerKM = Param(self.model.Vehicles, initialize=self._load_param(self.vehicles, 'VehicleID', 'Range'))
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
            depot_id = self.client_depot_map[c]
            # Obtener coordenadas del cliente
            client_row = self.clients[self.clients['ClientID'] == c].iloc[0]
            client_coords = (client_row['Latitude'], client_row['Longitude'])
            # Obtener coordenadas del depósito asociado al cliente
            # Asumiendo que tienes un DataFrame 'depots' con coordenadas
            depot_row = self.depots[self.depots['DepotID'] == depot_id].iloc[0]
            depot_coords = (depot_row['Latitude'], depot_row['Longitude'])
            # Calcular distancia euclidiana
            distances[c] = np.sqrt(
                (client_coords[0] - depot_coords[0])**2 +
                (client_coords[1] - depot_coords[1])**2
            )
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
        # Cada cliente debe ser servido por exactamente un vehículo
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



class Case1Base:
    def __init__(self, clients_file, drone_file, ev_file, gas_file, multi_vehicles_file, column_mapping):
        self.column_mapping = column_mapping

        # Cargar y mapear archivos
        self.clients = self._load_and_validate(
            clients_file,
            ['ClientID', 'DepotID', 'Demand', 'Longitude', 'Latitude'],  # 'Product' mapeado a 'Demand'
            self.column_mapping.get('clients', {})
        )
        self.drone_vehicles = self._load_and_validate(
            drone_file,
            ['VehicleID', 'Capacity', 'Range'],  # 'VehicleType' mapeado a 'VehicleID'
            self.column_mapping.get('vehicles', {})
        )
        self.ev_vehicles = self._load_and_validate(
            ev_file,
            ['VehicleID', 'Capacity', 'Range'],
            self.column_mapping.get('vehicles', {})
        )
        self.gas_vehicles = self._load_and_validate(
            gas_file,
            ['VehicleID', 'Capacity', 'Range'],
            self.column_mapping.get('vehicles', {})
        )
        self.multi_vehicles = self._load_and_validate(
            multi_vehicles_file,
            ['VehicleID', 'Capacity', 'Range'],
            self.column_mapping.get('vehicles', {})
        )

        # Combinar vehículos
        self.vehicles = pd.concat([self.drone_vehicles, self.ev_vehicles, self.gas_vehicles], ignore_index=True)

        # Los depósitos se obtienen de la columna 'DepotID' en 'Clients.csv'
        self.depots = pd.DataFrame({'DepotID': self.clients['DepotID'].unique()})

        # Diagnósticos
        print("Clients DataFrame Columns:", self.clients.columns.tolist())
        print("Depots DataFrame Columns:", self.depots.columns.tolist())
        print("Vehicles DataFrame Columns:", self.vehicles.columns.tolist())

        # Inicializar modelo
        self.model = ConcreteModel()
        self._initialize_sets()
        self._initialize_parameters()
        self._initialize_variables()
        self._define_objective()
        self._define_constraints()

    def _load_and_validate(self, file_path, required_columns, column_mapping):
        """Cargar un archivo CSV, aplicar mapeo de columnas y validar columnas requeridas."""
        df = pd.read_csv(file_path, encoding='utf-8')
        df.columns = df.columns.str.strip()
        df.rename(columns=column_mapping, inplace=True)
        if not set(required_columns).issubset(df.columns):
            raise ValueError(f"File {file_path} must contain the columns: {required_columns}. Found: {df.columns.tolist()}")
        return df

    def _initialize_sets(self):
        """Inicializar conjuntos."""
        self.model.Clients = Set(initialize=self.clients['ClientID'].tolist())
        self.model.Depots = Set(initialize=self.depots['DepotID'].tolist())
        self.model.Vehicles = Set(initialize=self.vehicles['VehicleID'].tolist())

    def _initialize_parameters(self):
        """Inicializar parámetros."""
        self.model.Demand = Param(
            self.model.Clients,
            initialize=self._load_param(self.clients, 'ClientID', 'Demand')
        )
        self.model.Capacity = Param(
            self.model.Vehicles,
            initialize=self._load_param(self.vehicles, 'VehicleID', 'Capacity')
        )
        self.model.CostPerKM = Param(
            self.model.Vehicles,
            initialize=self._load_param(self.vehicles, 'VehicleID', 'Range')  # Asumiendo 'Range' como costo
        )
        # Crear un parámetro que asocie cada cliente con su depósito
        self.client_depot_map = self.clients.set_index('ClientID')['DepotID'].to_dict()
        # Calcular distancias entre clientes y sus depósitos asociados
        self.model.Distance = Param(
            self.model.Clients,
            initialize=self._calculate_distances()
        )

    def _initialize_variables(self):
        """Inicializar variables de decisión."""
        self.model.x = Var(self.model.Clients, self.model.Vehicles, within=Binary)

    def _define_objective(self):
        """Definir objetivo."""
        self.model.obj = Objective(rule=self._objective_rule, sense=minimize)

    def _define_constraints(self):
        """Definir restricciones."""
        self.model.constraints = ConstraintList()
        self._add_base_constraints()

    def _load_param(self, dataframe, index_col, value_col):
        """Convertir columnas de DataFrame en un diccionario de parámetros Pyomo."""
        return dataframe.set_index(index_col)[value_col].to_dict()

    def _calculate_distances(self):
        """Calcular distancias entre clientes y sus depósitos asociados."""
        distances = {}
        for c in self.model.Clients:
            depot_id = self.client_depot_map[c]
            # Obtener coordenadas del cliente
            client_row = self.clients[self.clients['ClientID'] == c].iloc[0]
            client_coords = (client_row['Latitude'], client_row['Longitude'])
            # Suponiendo que los depósitos no tienen coordenadas, asignamos distancias aleatorias o predefinidas
            distances[c] = np.random.normal(loc=5, scale=1)  # Puedes ajustar esta lógica según tus necesidades
        return distances

    def _objective_rule(self, model):
        """Función objetivo para minimizar el costo total."""
        return sum(
            model.x[c, v] * model.Distance[c] * model.CostPerKM[v]
            for c in model.Clients for v in model.Vehicles
        )

    def _add_base_constraints(self):
        """Añadir restricciones básicas."""
        model = self.model
        # Cada cliente debe ser servido por exactamente un vehículo
        for c in model.Clients:
            model.constraints.add(
                sum(model.x[c, v] for v in model.Vehicles) == 1
            )
        # Respetar la capacidad de cada vehículo
        for v in model.Vehicles:
            model.constraints.add(
                sum(model.x[c, v] * model.Demand[c] for c in model.Clients) <= model.Capacity[v]
            )

    def solve(self, solver='glpk'):
        """Resolver el problema."""
        opt = SolverFactory(solver)
        results = opt.solve(self.model, tee=True)

        # Validar si la solución es factible
        if results.solver.termination_condition != TerminationCondition.optimal:
            print("\nWARNING: No feasible solution found. Check the constraints and inputs.")
            return results

        print("\nSolution found successfully.")
        return results

    def save_results(self, output_file):
        """Guardar resultados en un archivo CSV."""
        results = []
        for c in self.model.Clients:
            for v in self.model.Vehicles:
                val = self.model.x[c, v]()
                if val is not None and val > 0.5:
                    results.append({
                        'Client': c,
                        'Depot': self.client_depot_map[c],  # Obtener el depósito asociado al cliente
                        'Vehicle': v
                    })
        if not results:
            print("No valid assignments found. Check model constraints and inputs.")
        else:
            pd.DataFrame(results).to_csv(output_file, index=False)


# Configuración de mapeo dinámico
case_1_mapping = {
    'clients': {'Product': 'Demand'},
    'vehicles': {'VehicleType': 'VehicleID'},
    'depots': {}  # Sin cambios
}

case1 = Case1Base(
    clients_file=r"C:\Users\cpenu\Documents\6-semestre\MoS\Proyecto2-MoS\case_1_base\case_1_base\Clients.csv",       
    drone_file=r"C:\Users\cpenu\Documents\6-semestre\MoS\Proyecto2-MoS\case_1_base\case_1_base\drone_only.csv",        
    ev_file=r"C:\Users\cpenu\Documents\6-semestre\MoS\Proyecto2-MoS\case_1_base\case_1_base\ev_only.csv",           
    gas_file=r"C:\Users\cpenu\Documents\6-semestre\MoS\Proyecto2-MoS\case_1_base\case_1_base\gas_car_only.csv",          
    multi_vehicles_file=r"C:\Users\cpenu\Documents\6-semestre\MoS\Proyecto2-MoS\case_1_base\case_1_base\multi_vehicles.csv",
    column_mapping=case_1_mapping
)
case1.solve()
case1.save_results("case_1_base_results.csv")

