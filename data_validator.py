"""
Data validation and preprocessing utilities for the dropout prediction model
"""

import numpy as np
import pandas as pd
from collections import defaultdict


class DataValidator:
    """Validates and cleans input data for prediction"""

    # Define expected value ranges and categories
    FIELD_CONFIGS = {
        # Categorical fields with expected values
        "School": {
            "type": "categorical",
            "description": "School code",
            "default": "GP",
        },
        "Gender": {
            "type": "categorical",
            "description": "Student gender",
            "valid_values": ["M", "F"],
            "default": "M",
        },
        "Address": {
            "type": "categorical",
            "description": "Student address type",
            "valid_values": ["U", "R"],
            "default": "U",
        },
        "Family_Size": {
            "type": "categorical",
            "description": "Family size category",
            "valid_values": ["LE3", "GT3"],
            "default": "LE3",
        },
        "Parental_Status": {
            "type": "categorical",
            "description": "Parental marital status",
            "valid_values": ["together", "apart", "widowed", "divorced", "single"],
            "default": "together",
        },
        "Mother_Job": {
            "type": "categorical",
            "description": "Mother's job",
            "default": "other",
        },
        "Father_Job": {
            "type": "categorical",
            "description": "Father's job",
            "default": "other",
        },
        "Guardian": {
            "type": "categorical",
            "description": "Guardian type",
            "default": "mother",
        },
        "School_Support": {
            "type": "categorical",
            "description": "School support",
            "valid_values": ["yes", "no"],
            "default": "no",
        },
        "Family_Support": {
            "type": "categorical",
            "description": "Family support",
            "valid_values": ["yes", "no"],
            "default": "no",
        },
        "Extra_Paid_Class": {
            "type": "categorical",
            "description": "Extra paid classes",
            "valid_values": ["yes", "no"],
            "default": "no",
        },
        "Extra_Curricular_Activities": {
            "type": "categorical",
            "description": "Extracurricular activities",
            "valid_values": ["yes", "no"],
            "default": "no",
        },
        "Attended_Nursery": {
            "type": "categorical",
            "description": "Attended nursery",
            "valid_values": ["yes", "no"],
            "default": "yes",
        },
        "Wants_Higher_Education": {
            "type": "categorical",
            "description": "Wants higher education",
            "valid_values": ["yes", "no"],
            "default": "yes",
        },
        "Internet_Access": {
            "type": "categorical",
            "description": "Internet access",
            "valid_values": ["yes", "no"],
            "default": "yes",
        },
        "In_Relationship": {
            "type": "categorical",
            "description": "In a relationship",
            "valid_values": ["yes", "no"],
            "default": "no",
        },
        "Reason_for_Choosing_School": {
            "type": "categorical",
            "description": "Reason for choosing school",
            "default": "reputation",
        },
        # Numerical fields with valid ranges
        "Age": {
            "type": "numeric",
            "description": "Student age",
            "min": 15,
            "max": 25,
            "default": 17,
        },
        "Mother_Education": {
            "type": "numeric",
            "description": "Mother's education level",
            "min": 1,
            "max": 4,
            "default": 2,
        },
        "Father_Education": {
            "type": "numeric",
            "description": "Father's education level",
            "min": 1,
            "max": 4,
            "default": 2,
        },
        "Travel_Time": {
            "type": "numeric",
            "description": "Travel time in hours",
            "min": 1,
            "max": 4,
            "default": 2,
        },
        "Study_Time": {
            "type": "numeric",
            "description": "Study time per week in hours",
            "min": 1,
            "max": 4,
            "default": 2,
        },
        "Number_of_Failures": {
            "type": "numeric",
            "description": "Number of past class failures",
            "min": 0,
            "max": 4,
            "default": 0,
        },
        "Family_Relationship": {
            "type": "numeric",
            "description": "Family relationship quality",
            "min": 1,
            "max": 5,
            "default": 3,
        },
        "Free_Time": {
            "type": "numeric",
            "description": "Free time after school",
            "min": 1,
            "max": 5,
            "default": 3,
        },
        "Going_Out": {
            "type": "numeric",
            "description": "Frequency of going out",
            "min": 1,
            "max": 5,
            "default": 3,
        },
        "Weekend_Alcohol_Consumption": {
            "type": "numeric",
            "description": "Weekend alcohol consumption",
            "min": 1,
            "max": 5,
            "default": 1,
        },
        "Weekday_Alcohol_Consumption": {
            "type": "numeric",
            "description": "Weekday alcohol consumption",
            "min": 1,
            "max": 5,
            "default": 1,
        },
        "Health_Status": {
            "type": "numeric",
            "description": "Current health status",
            "min": 1,
            "max": 5,
            "default": 3,
        },
        "Number_of_Absences": {
            "type": "numeric",
            "description": "Number of absences",
            "min": 0,
            "max": 93,
            "default": 0,
        },
        "Grade_1": {
            "type": "numeric",
            "description": "First period grade",
            "min": 0,
            "max": 20,
            "default": 10,
        },
        "Grade_2": {
            "type": "numeric",
            "description": "Second period grade",
            "min": 0,
            "max": 20,
            "default": 10,
        },
        "Final_Grade": {
            "type": "numeric",
            "description": "Final grade",
            "min": 0,
            "max": 20,
            "default": 10,
        },
    }

    @staticmethod
    def validate_field(field_name, value):
        """
        Validate and clean a single field
        Returns: (cleaned_value, is_valid, error_message)
        """
        if field_name not in DataValidator.FIELD_CONFIGS:
            return value, False, f"Unknown field: {field_name}"

        config = DataValidator.FIELD_CONFIGS[field_name]

        try:
            if config["type"] == "categorical":
                # Convert to string and handle empty values
                str_value = str(value).strip().lower()

                if not str_value or str_value in ["none", "nan", "unknown", ""]:
                    return config.get("default", "unknown"), True, None

                # If specific valid values are defined, validate
                if "valid_values" in config:
                    valid_values = [v.lower() for v in config["valid_values"]]
                    if str_value in valid_values:
                        return str_value, True, None
                    else:
                        # Return default for invalid categorical value
                        return (
                            config.get("default", valid_values[0]),
                            True,
                            f"Warning: Unknown value for {field_name}, using default '{config.get('default', valid_values[0])}'",
                        )
                else:
                    # For unconstrained categorical fields, just return the value
                    # (will be handled by label encoder during prediction)
                    return str_value, True, None

            elif config["type"] == "numeric":
                # Convert to numeric
                try:
                    numeric_value = float(value)
                except (ValueError, TypeError):
                    return (
                        config.get("default", 0),
                        True,
                        f"Warning: Could not convert {field_name}={value} to number, using default {config.get('default', 0)}",
                    )

                # Check range
                min_val = config.get("min", float("-inf"))
                max_val = config.get("max", float("inf"))

                if numeric_value < min_val:
                    clamped = min_val
                    return (
                        clamped,
                        True,
                        f"Warning: {field_name}={numeric_value} below minimum {min_val}, clamped to {min_val}",
                    )
                elif numeric_value > max_val:
                    clamped = max_val
                    return (
                        clamped,
                        True,
                        f"Warning: {field_name}={numeric_value} above maximum {max_val}, clamped to {max_val}",
                    )
                else:
                    return numeric_value, True, None

        except Exception as e:
            return (
                config.get("default", 0),
                True,
                f"Warning: Error validating {field_name}: {str(e)}, using default {config.get('default', 0)}",
            )

        return value, False, "Unknown error"

    @staticmethod
    def validate_student_data(student_data):
        """
        Validate all student data fields
        Returns: (cleaned_data, warnings_list)
        """
        cleaned_data = {}
        warnings = []

        for field_name, value in student_data.items():
            cleaned_value, is_valid, warning = DataValidator.validate_field(
                field_name, value
            )
            cleaned_data[field_name] = cleaned_value
            if warning:
                warnings.append(warning)

        return cleaned_data, warnings


class UnknownCategoryHandler:
    """Handles unknown categories during prediction"""

    def __init__(self, label_encoders, categorical_cols):
        """
        Initialize with label encoders from training

        Args:
            label_encoders: Dict of LabelEncoders from training
            categorical_cols: List of categorical column names
        """
        self.label_encoders = label_encoders
        self.categorical_cols = categorical_cols
        self.unknown_mappings = defaultdict(lambda: 0)  # Default to 0 for unknown

        # Build reverse mappings for known values
        self.known_values = {}
        for col, le in label_encoders.items():
            self.known_values[col] = set(le.classes_)

    def encode_categorical(self, feature_name, value):
        """
        Encode categorical feature, handling unknown values

        Returns: (encoded_value, warning_message or None)
        """
        if feature_name not in self.label_encoders:
            return 0, f"Feature {feature_name} not found in label encoders"

        le = self.label_encoders[feature_name]
        str_value = str(value).strip()

        # Check if value is known
        if str_value in le.classes_:
            return le.transform([str_value])[0], None
        else:
            # Unknown value - return 0 (safest default)
            return 0, f"Unknown value for {feature_name}='{str_value}', encoded as 0"

    def get_all_known_values(self, feature_name):
        """Get all known values for a categorical feature"""
        if feature_name in self.label_encoders:
            return list(self.label_encoders[feature_name].classes_)
        return []
