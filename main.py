# ============================================
# Cybersecurity Resolution-Time Prediction API
# ============================================

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# --------------------------------------------
# Load model files
# --------------------------------------------

BASE_DIRECTORY = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIRECTORY / "portable_model.json"
OPTIONS_PATH = BASE_DIRECTORY / "input_options.json"


def load_json_file(file_path: Path) -> dict:
    if not file_path.exists():
        raise RuntimeError(
            f"Required file was not found: {file_path.name}"
        )

    with file_path.open(
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


portable_model = load_json_file(MODEL_PATH)
input_options = load_json_file(OPTIONS_PATH)


categorical_features = portable_model[
    "categorical_features"
]

numerical_features = portable_model[
    "numerical_features"
]

category_labels = portable_model[
    "category_labels"
]

encoded_sizes = portable_model[
    "encoded_sizes"
]

standard_deviations = portable_model[
    "scaler_standard_deviations"
]

regression_coefficients = portable_model[
    "regression_coefficients"
]

regression_intercept = float(
    portable_model["regression_intercept"]
)


# --------------------------------------------
# Create FastAPI application
# --------------------------------------------

app = FastAPI(
    title="Cybersecurity Incident Resolution API",
    description=(
        "Predicts cybersecurity incident resolution "
        "time using the Linear Regression model "
        "trained with Apache Spark MLlib."
    ),
    version="1.0.0"
)


# Allow the Streamlit frontend to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# --------------------------------------------
# Request and response formats
# --------------------------------------------

class IncidentInput(BaseModel):
    country: str

    year: int = Field(
        ge=2015,
        le=2024
    )

    attack_type: str
    target_industry: str

    financial_loss_million: float = Field(
        ge=0
    )

    number_of_affected_users: int = Field(
        ge=0
    )

    attack_source: str
    security_vulnerability_type: str
    defense_mechanism_used: str


class PredictionResponse(BaseModel):
    predicted_resolution_time_hours: float
    model_name: str


# Match API names with the original dataset names.
categorical_field_mapping = {
    "Country": "country",
    "Attack Type": "attack_type",
    "Target Industry": "target_industry",
    "Attack Source": "attack_source",
    "Security Vulnerability Type": (
        "security_vulnerability_type"
    ),
    "Defense Mechanism Used": (
        "defense_mechanism_used"
    )
}


# --------------------------------------------
# Prediction functions
# --------------------------------------------

def model_to_dictionary(
    incident: IncidentInput
) -> dict:

    if hasattr(incident, "model_dump"):
        return incident.model_dump()

    return incident.dict()


def validate_categorical_values(
    incident_data: dict
) -> None:

    for feature_name in categorical_features:

        api_field_name = categorical_field_mapping[
            feature_name
        ]

        provided_value = incident_data[
            api_field_name
        ]

        allowed_values = input_options.get(
            feature_name,
            category_labels[feature_name]
        )

        if provided_value not in allowed_values:
            raise HTTPException(
                status_code=422,
                detail={
                    "invalid_field": api_field_name,
                    "provided_value": provided_value,
                    "allowed_values": allowed_values
                }
            )


def encode_category(
    feature_name: str,
    category_value: str
) -> list[float]:

    labels = category_labels[feature_name]

    category_index = labels.index(
        category_value
    )

    vector_size = int(
        encoded_sizes[feature_name]
    )

    encoded_vector = [
        0.0
    ] * vector_size

    # Spark OneHotEncoder represents its final
    # dropped category using an all-zero vector.
    if category_index < vector_size:
        encoded_vector[
            category_index
        ] = 1.0

    return encoded_vector


def calculate_prediction(
    incident_data: dict
) -> float:

    year_index = float(
        incident_data["year"] - 2015
    )

    financial_loss = float(
        incident_data[
            "financial_loss_million"
        ]
    )

    affected_users = float(
        incident_data[
            "number_of_affected_users"
        ]
    )

    if affected_users > 0:
        loss_per_1000_users = (
            financial_loss
            / affected_users
        ) * 1000
    else:
        loss_per_1000_users = 0.0


    numerical_values = {
        "Year Index": year_index,

        "Financial Loss (in Million $)": (
            financial_loss
        ),

        "Number of Affected Users": (
            affected_users
        ),

        "Loss per 1000 Users": (
            loss_per_1000_users
        )
    }


    categorical_values = {
        feature_name: incident_data[
            categorical_field_mapping[
                feature_name
            ]
        ]
        for feature_name in categorical_features
    }


    # Recreate the exact feature order used by Spark.
    assembled_features = [
        float(
            numerical_values[feature_name]
        )
        for feature_name in numerical_features
    ]

    for feature_name in categorical_features:

        assembled_features.extend(
            encode_category(
                feature_name,
                categorical_values[feature_name]
            )
        )


    expected_feature_count = len(
        regression_coefficients
    )

    if len(assembled_features) != expected_feature_count:
        raise RuntimeError(
            "Feature-vector size does not match "
            "the trained model."
        )


    # Apply the same StandardScaler transformation.
    scaled_features = [
        (
            feature_value / standard_deviation
            if standard_deviation != 0
            else 0.0
        )
        for feature_value, standard_deviation
        in zip(
            assembled_features,
            standard_deviations
        )
    ]


    prediction = (
        regression_intercept
        + sum(
            coefficient * feature_value

            for coefficient, feature_value
            in zip(
                regression_coefficients,
                scaled_features
            )
        )
    )

    return float(prediction)


# --------------------------------------------
# API routes
# --------------------------------------------

@app.get("/")
def home():
    return {
        "message": (
            "Cybersecurity Incident Resolution "
            "Prediction API"
        ),
        "documentation": "/docs"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_name": portable_model["model_name"],
        "feature_count": len(
            regression_coefficients
        )
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict_resolution_time(
    incident: IncidentInput
):

    incident_data = model_to_dictionary(
        incident
    )

    validate_categorical_values(
        incident_data
    )

    predicted_hours = calculate_prediction(
        incident_data
    )

    return {
        "predicted_resolution_time_hours": round(
            predicted_hours,
            2
        ),
        "model_name": portable_model[
            "model_name"
        ]
    }