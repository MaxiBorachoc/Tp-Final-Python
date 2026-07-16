from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_STATE = 42
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "autos_usados.csv"

MODELOS_POR_MARCA = {
    "Toyota": ["Corolla", "Etios", "Yaris", "Hilux"],
    "Volkswagen": ["Gol", "Polo", "Virtus", "Amarok"],
    "Ford": ["Fiesta", "Focus", "Ranger", "EcoSport"],
    "Chevrolet": ["Onix", "Cruze", "Tracker", "S10"],
    "Renault": ["Clio", "Sandero", "Logan", "Duster"],
    "Fiat": ["Cronos", "Argo", "Mobi", "Toro"],
    "Peugeot": ["208", "308", "Partner", "2008"],
}

VALOR_MARCA = {
    "Toyota": 2600,
    "Volkswagen": 1900,
    "Ford": 1700,
    "Chevrolet": 1500,
    "Renault": 1000,
    "Fiat": 900,
    "Peugeot": 1200,
}

VALOR_MODELO = {
    "Hilux": 6500,
    "Amarok": 6000,
    "Ranger": 5600,
    "S10": 5200,
    "Toro": 3800,
    "Tracker": 3200,
    "Duster": 2700,
    "EcoSport": 2400,
    "Corolla": 3100,
    "Cruze": 2600,
    "Virtus": 2200,
    "Polo": 1800,
    "Yaris": 1600,
    "208": 1500,
    "308": 1300,
    "2008": 2300,
    "Onix": 1100,
    "Cronos": 950,
    "Argo": 700,
    "Mobi": -800,
    "Gol": 400,
    "Fiesta": 350,
    "Focus": 1300,
    "Etios": 850,
    "Clio": -400,
    "Sandero": 250,
    "Logan": 200,
    "Partner": 900,
}

VALOR_COMBUSTIBLE = {
    "Nafta": 0,
    "Diesel": 1700,
    "Hibrido": 3200,
}

VALOR_TRANSMISION = {
    "Manual": 0,
    "Automatica": 1800,
}


def generar_dataset(cantidad: int = 420) -> pd.DataFrame:
    """Genera un dataset sintetico y reproducible de autos usados."""
    rng = np.random.default_rng(RANDOM_STATE)
    marcas = np.array(list(MODELOS_POR_MARCA.keys()))
    combustibles = np.array(list(VALOR_COMBUSTIBLE.keys()))
    transmisiones = np.array(list(VALOR_TRANSMISION.keys()))

    registros = []
    for _ in range(cantidad):
        marca = rng.choice(marcas)
        modelo = rng.choice(MODELOS_POR_MARCA[marca])
        anio = int(rng.integers(2005, 2025))
        edad = 2026 - anio
        kilometraje_base = edad * rng.normal(11500, 2600)
        kilometraje = int(max(0, kilometraje_base + rng.normal(0, 18000)))
        combustible = rng.choice(combustibles, p=[0.72, 0.22, 0.06])
        transmision = rng.choice(transmisiones, p=[0.63, 0.37])

        precio = (
            24500
            + VALOR_MARCA[marca]
            + VALOR_MODELO[modelo]
            + VALOR_COMBUSTIBLE[combustible]
            + VALOR_TRANSMISION[transmision]
            - edad * 950
            - kilometraje * 0.045
            + rng.normal(0, 1200)
        )
        precio = round(float(max(2500, precio)), 2)

        registros.append(
            {
                "marca": marca,
                "modelo": modelo,
                "anio": anio,
                "kilometraje": kilometraje,
                "combustible": combustible,
                "transmision": transmision,
                "precio_usd": precio,
            }
        )

    return pd.DataFrame(registros)


def main() -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset = generar_dataset()
    dataset.to_csv(DATA_PATH, index=False)
    print(f"Dataset generado: {DATA_PATH}")
    print(f"Registros: {len(dataset)}")


if __name__ == "__main__":
    main()
