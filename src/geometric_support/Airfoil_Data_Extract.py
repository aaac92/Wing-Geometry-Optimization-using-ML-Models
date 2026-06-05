import matplotlib.pyplot as plt
import csv
import numpy as np
from tqdm import tqdm

class Airfoil_Tools_Exctract:
        
    def __init__(self):
        self.file = str() # File source
        self.name = str()
        self.C = None
        self.X_UPPER, self.Z_UPPER = [], []
        self.X_LOWER, self.Z_LOWER = [], []

    def airfoil_coords(self, verbose: bool = True):
        '''
        Reads airfoil coordinates from an Airfoiltools.com CSV file.
        A verbose flag is included to toggle the tqdm progress bar.
        '''
        # Reading from Airfoiltools.com CSV
        file_path = self.file
        with open(file_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            lines = list(reader)

            # Extract Airfoil name
            name_start = lines[0]
            if name_start[0].startswith("Name"):
                self.name = str(name_start[1])

            # Extracts chord length
            chord_line = lines[1]
            if chord_line[0].startswith("Chord(mm)"):
                self.C = float(chord_line[1])
            # Airfoil coordinate search
            for i, line in enumerate(lines):
                if line and line[0].strip() == "Airfoil surface":
                    coord_start = i + 2 
                    break
            else:
                raise ValueError("Could not find 'Airfoil surface' marker in file")

            # Coordinate Extraction
            surface = 'upper'
            tot_coords = len(lines[coord_start:])
            
            # Added 'disable=not verbose' to conditionally hide the progress bar
            airfoil_pbar = tqdm(total=tot_coords, desc='Extracting ' + self.name + '\'s coordinates: ', disable=not verbose)
            
            for i, row in enumerate(lines[coord_start:]):
                airfoil_pbar.update(1)
                if len(row) < 2:
                    continue
                try:
                    x = float(row[0])
                    z = float(row[1])
                except ValueError:
                    airfoil_pbar.update(tot_coords - (i + 1))
                    airfoil_pbar.close() 
                    break  # Break when finding non-numeric values like:"Camber Line"

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
                airfoil_pbar.close() 
                

    def ate_verif(self):
        # Result Verification
        print(f"Chord length: {self.C} mm")
        print(f"Upper surface points: {len(self.X_UPPER)}")
        print(f"Lower surface points: {len(self.X_LOWER)}")
        print('X UPPER: ' + str(self.X_UPPER))
        print('Z UPPER: ' + str(self.Z_UPPER))
        print('X LOWER: ' + str(self.X_LOWER))
        print('Z LOWER: ' + str(self.Z_LOWER))

    def ate_plt(self):
        # Plots extracted airfoil
        plt.scatter(self.X_UPPER, self.Z_UPPER, label = "Upper Surface", color = 'blue')
        plt.scatter(self.X_LOWER, self.Z_LOWER, label = "Lower Surface", color = 'red')
        plt.legend('Airfoil: ' + self.name)
        plt.axis('equal')
        plt.show()

    def ate_upscaling(self, x_res, n = 15):
        # This function can "upscale" an airfoil if needed for obtaining better CST parameters

        pol_coeff_foil_u = np.polyfit(self.X_UPPER, self.Z_UPPER, n)
        poly_Airfoil_u = np.poly1d(pol_coeff_foil_u)
        
        pol_coeff_foil_l = np.polyfit(self.X_LOWER, self.Z_LOWER, n)
        poly_Airfoil_l = np.poly1d(pol_coeff_foil_l)

        up_X_UPPER = np.linspace(self.X_UPPER[0], self.X_UPPER[-1], x_res)
        up_X_LOWER = np.linspace(self.X_LOWER[0], self.X_LOWER[-1], x_res)
        up_Z_UPPER = poly_Airfoil_u(up_X_UPPER)
        up_Z_LOWER = poly_Airfoil_l(up_X_LOWER)

        return(up_X_UPPER, up_X_LOWER, up_Z_UPPER, up_Z_LOWER)
