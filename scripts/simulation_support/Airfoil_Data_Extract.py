import matplotlib.pyplot as plt
import csv
import numpy as np
from tqdm import tqdm

class Airfoil_Tools_Exctract:
        
    def __init__(self):
        # Ubicación (carpeta raiz del código en este caso) y nombre del archivo.
        self.file= file = str()

        # Initialización de listas para coordenadas y variable para la cuerda
        self.name = name = str()
        self.C = C = None
        self.X_UPPER, self.Z_UPPER = X_UPPER, Z_UPPER = [], []
        self.X_LOWER, self.Z_LOWER = X_LOWER, Z_LOWER = [], []

    def airfoil_coords(self):
        # Lectura de información
        file_path = self.file
        with open(file_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile) #Almacena info del csv
            lines = list(reader) #Convierte lineas del CSV en una lista

            # Extracción del Nombre
            name_start = lines[0]
            if name_start[0].startswith("Name"):
                self.name = str(name_start[1])

            # Extrae la longitud de la cuerda de la linea 2 (index 1)
            chord_line = lines[1]
            if chord_line[0].startswith("Chord(mm)"):
                self.C = float(chord_line[1])  # Extrae unicamente el valor numerico después de la coma
            # Buscamos las coordenadas del perfil
            for i, line in enumerate(lines):
                if line and line[0].strip() == "Airfoil surface":
                    coord_start = i + 2  # Las coordenadas inician 2 lineas despues 
                    break
            else:
                raise ValueError("Could not find 'Airfoil surface' marker in file")

            # Extracción de coordenadas
            surface = 'upper'
            tot_coords = len(lines[coord_start:])
            airfoil_pbar = tqdm(total = tot_coords, desc= 'Extracting ' + self.name + '\'s coordinates: ')
            for i, row in enumerate(lines[coord_start:]):
                airfoil_pbar.update(1)
                if len(row) < 2:
                    continue
                try:
                    x = float(row[0])
                    z = float(row[1])
                except ValueError:
                    airfoil_pbar.update(tot_coords - (i + 1)) #Fills remaining steps of progress bar
                    airfoil_pbar.close #Closes progress bar if the break is triggered
                    break  # Rompe condicional al encontrar valores no numericos como:"Camber Line"

                if surface == 'upper':
                    self.X_UPPER.append(x)
                    self.Z_UPPER.append(z)
                    if x == 0:
                        surface = 'lower'
                        self.X_LOWER.append(x)
                        self.Z_LOWER.append(z)
                else:
                    self.X_LOWER.append(x)
                    self.Z_LOWER.append(z)
            else:
                airfoil_pbar.close #Closes progress bar
                

    def ate_verif(self):
        # Verificación de Resultados
        print(f"Chord length: {self.C} mm")
        print(f"Upper surface points: {len(self.X_UPPER)}")
        print(f"Lower surface points: {len(self.X_LOWER)}")
        print('X UPPER: ' + str(self.X_UPPER))
        print('Z UPPER: ' + str(self.Z_UPPER))
        print('X LOWER: ' + str(self.X_LOWER))
        print('Z LOWER: ' + str(self.Z_LOWER))

    def ate_plt(self):
        #Grafica de los puntos geometricos del perfil
        plt.scatter(self.X_UPPER, self.Z_UPPER, label = "Upper Surface", color = 'blue')
        plt.scatter(self.X_LOWER, self.Z_LOWER, label = "Lower Surface", color = 'red')
        plt.legend('Airfoil: ' + self.name)
        plt.axis('equal')
        plt.show()

    def ate_upscaling(self, x_res, n = 15):
        # Genera mas puntos en caso de tener pocos para evitar errores usando el metodo CST
        #Encontrar el polinomio de regresion del perfil base para superficie superior
        pol_coeff_foil_u = np.polyfit(self.X_UPPER, self.Z_UPPER, n) #Regresa coeficientes del polinomio que se ajusta a los puntos con el grado determinado
        poly_Airfoil_u = np.poly1d(pol_coeff_foil_u) #Construye un polinomio mas "normal" con los valores de la lista como coeficientes lista[0]*x^n + lista[1]*x^n-1 ...
        
        #Encontrar el polinomio de regresion del perfil base para superficie inferior
        pol_coeff_foil_l = np.polyfit(self.X_LOWER, self.Z_LOWER, n)
        poly_Airfoil_l = np.poly1d(pol_coeff_foil_l)

        up_X_UPPER = np.linspace(self.X_UPPER[0], self.X_UPPER[-1], x_res)
        up_X_LOWER = np.linspace(self.X_LOWER[0], self.X_LOWER[-1], x_res)
        up_Z_UPPER = poly_Airfoil_u(up_X_UPPER)
        up_Z_LOWER = poly_Airfoil_l(up_X_LOWER)

        return(up_X_UPPER, up_X_LOWER, up_Z_UPPER, up_Z_LOWER)
        #return(poly_Airfoil_u, poly_Airfoil_l)
