from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout, update_session_auth_hash
from django.contrib.auth.forms import PasswordChangeForm
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db import models
from django.utils import timezone
from datetime import timedelta
import os
import numpy as np
import random
from django.core.mail import send_mail
from .models import DropoutPrediction, PasswordResetOTP, ContactMessage
from data_validator import DataValidator, UnknownCategoryHandler


# Helper function to format last login in user-friendly format
def format_last_login(last_login):
    """Format last login datetime in a user-friendly way"""
    if last_login is None:
        return "Never logged in"

    now = timezone.now()
    diff = now - last_login

    # Handle different time differences
    if diff.days == 0:
        # Same day
        hours = diff.seconds // 3600
        minutes = (diff.seconds % 3600) // 60

        if hours == 0:
            if minutes == 0:
                return "Just now"
            elif minutes == 1:
                return "1 minute ago"
            else:
                return f"{minutes} minutes ago"
        elif hours == 1:
            return "1 hour ago"
        else:
            return f"{hours} hours ago"
    elif diff.days == 1:
        # Yesterday
        time_str = last_login.strftime("%I:%M %p")  # e.g., "02:30 PM"
        return f"Yesterday at {time_str}"
    elif diff.days < 7:
        # Within a week
        return f"{diff.days} days ago"
    else:
        # More than a week ago - show the actual date
        return last_login.strftime("%b %d, %Y")  # e.g., "Mar 25, 2026"


# Home Page View
def home(request):
    return render(request, "home.html")


# Login View
def login_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            messages.success(request, f"Welcome back, {username}!")
            return redirect("home")
        else:
            messages.error(request, "Invalid username or password.")

    return render(request, "login.html")


# Register View
def register_view(request):
    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        email = request.POST.get("email", "").strip()
        password = request.POST.get("password")
        password2 = request.POST.get("password2")

        # Basic validations
        if not username:
            messages.error(request, "Username is required.")
            return render(request, "register.html")

        # Check if passwords match
        if password != password2:
            messages.error(request, "Passwords do not match.")
            return render(request, "register.html")

        # Check if username already exists
        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already in use.")
            return render(request, "register.html")

        # Check if email already exists
        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already in use.")
            return render(request, "register.html")

        # Create new user
        try:
            User.objects.create_user(username=username, email=email, password=password)
            messages.success(
                request,
                f"Welcome to Aetheria, {username}! Your account was created successfully. Please log in.",
            )
            return redirect("login")
        except Exception as e:
            messages.error(request, f"Error creating account: {str(e)}")

    return render(request, "register.html")


# Logout View
def logout_view(request):
    logout(request)
    messages.success(request, "You have been logged out successfully.")
    return redirect("home")


# About Us View
def about(request):
    return render(request, "about.html")


# Contact Us View
def contact(request):
    if request.method == "POST":
        name = request.POST.get("name")
        email = request.POST.get("email")
        subject = request.POST.get("subject")
        message = request.POST.get("message")

        # Save the message to the database
        from .models import ContactMessage

        ContactMessage.objects.create(
            name=name, email=email, subject=subject, message=message
        )

        messages.success(
            request, "Thank you for contacting us! We will get back to you soon."
        )
        return redirect("contact")

    return render(request, "contact.html")


# User Dashboard View
@login_required(login_url="login")
def dashboard(request):
    user = request.user

    # Get prediction statistics
    user_predictions = DropoutPrediction.objects.filter(user=user)
    total_predictions = user_predictions.count()
    high_risk_predictions = user_predictions.filter(predicted_dropout=True).count()
    low_risk_predictions = user_predictions.filter(predicted_dropout=False).count()

    # Calculate percentages
    high_risk_percentage = 0
    low_risk_percentage = 0
    if total_predictions > 0:
        high_risk_percentage = round(
            (high_risk_predictions / total_predictions) * 100, 1
        )
        low_risk_percentage = round((low_risk_predictions / total_predictions) * 100, 1)

    # Calculate average risk if predictions exist
    avg_risk = 0
    if total_predictions > 0:
        avg_risk = round(
            user_predictions.aggregate(models.Avg("dropout_risk_percentage"))[
                "dropout_risk_percentage__avg"
            ]
            or 0,
            1,
        )

    context = {
        "user_info": {
            "username": user.username,
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "date_joined": user.date_joined,
            "last_login": format_last_login(user.last_login),
        },
        "prediction_stats": {
            "total_predictions": total_predictions,
            "high_risk_count": high_risk_predictions,
            "low_risk_count": low_risk_predictions,
            "average_risk": avg_risk,
            "high_risk_percentage": high_risk_percentage,
            "low_risk_percentage": low_risk_percentage,
        },
        "chart_data": {
            "students": list(
                user_predictions.values(
                    "student_name",
                    "dropout_risk_percentage",
                    "school",
                    "grade_1",
                    "grade_2",
                    "final_grade",
                    "number_of_absences",
                    "predicted_dropout",
                    "study_time",
                    "age",
                )
            ),
        },
    }
    return render(request, "dashboard.html", context)


# Load ML Model artifacts
def load_ml_model():
    """Load trained model and preprocessing artifacts"""
    try:
        import joblib
        import pickle
    except ImportError:
        print("Error: required modules not installed")
        return None, None, None, None, None

    ml_dir = os.path.join(os.path.dirname(__file__), "..", "ml_models")

    try:
        model = joblib.load(os.path.join(ml_dir, "dropout_model.joblib"))
        label_encoders = joblib.load(os.path.join(ml_dir, "label_encoders.joblib"))
        feature_names = joblib.load(os.path.join(ml_dir, "feature_names.joblib"))
        column_info = joblib.load(os.path.join(ml_dir, "column_info.joblib"))

        # Try loading scaler from pickle file first (new format), fallback to joblib (old format)
        scaler = None
        scaler_pkl = os.path.join(ml_dir, "scaler.pkl")
        scaler_joblib = os.path.join(ml_dir, "scaler.joblib")

        if os.path.exists(scaler_pkl):
            with open(scaler_pkl, "rb") as f:
                scaler = pickle.load(f)
        elif os.path.exists(scaler_joblib):
            scaler = joblib.load(scaler_joblib)
        else:
            print("Error: Scaler file not found")
            return None, None, None, None, None

        # Validate that all artifacts loaded correctly
        if (
            model is None
            or scaler is None
            or feature_names is None
            or column_info is None
        ):
            print("Error: One or more model artifacts failed to load properly")
            return None, None, None, None, None

        # Validate that scaler is fitted
        if not hasattr(scaler, "mean_") or scaler.mean_ is None:
            print("Error: Scaler is not fitted. Retraining required.")
            return None, None, None, None, None

        return model, label_encoders, scaler, feature_names, column_info
    except Exception as e:
        print(f"Error loading ML model artifacts: {str(e)}")
        import traceback

        traceback.print_exc()
        return None, None, None, None, None


# Predict Dropout Risk
@login_required(login_url="login")
def predict_dropout_risk(request):
    """Student dropout risk prediction calculator with improved data validation"""

    model, label_encoders, scaler, feature_names, column_info = load_ml_model()

    if model is None or scaler is None or feature_names is None or column_info is None:
        messages.error(request, "Model not available. Please contact administrator.")
        return redirect("home")

    prediction_result = None
    validation_warnings = []

    if request.method == "POST":
        try:
            # Extract age first for validation
            age_val = request.POST.get("age", "0")
            try:
                age = int(age_val)
            except ValueError:
                age = 0

            if age >= 18:
                messages.error(request, "Age must be below 18 years old.")
                return render(request, "predict.html")

            # Extract all form data
            raw_student_data = {
                "Student_Name": request.POST.get("student_name", ""),
                "School": request.POST.get("school", ""),
                "Gender": request.POST.get("gender", ""),
                "Age": age,
                "Address": request.POST.get("address", ""),
                "Family_Size": request.POST.get("family_size", ""),
                "Parental_Status": request.POST.get("parental_status", ""),
                "Mother_Education": request.POST.get("mother_education", "0"),
                "Father_Education": request.POST.get("father_education", "0"),
                "Mother_Job": request.POST.get("mother_job", ""),
                "Father_Job": request.POST.get("father_job", ""),
                "Reason_for_Choosing_School": request.POST.get(
                    "reason_choosing_school", ""
                ),
                "Guardian": request.POST.get("guardian", ""),
                "Travel_Time": request.POST.get("travel_time", "0"),
                "Study_Time": request.POST.get("study_time", "0"),
                "Number_of_Failures": request.POST.get("failures", "0"),
                "School_Support": request.POST.get("school_support", ""),
                "Family_Support": request.POST.get("family_support", ""),
                "Extra_Paid_Class": request.POST.get("extra_paid_class", ""),
                "Extra_Curricular_Activities": request.POST.get("extra_curricular", ""),
                "Attended_Nursery": request.POST.get("attended_nursery", ""),
                "Wants_Higher_Education": request.POST.get("higher_education", ""),
                "Internet_Access": request.POST.get("internet_access", ""),
                "In_Relationship": request.POST.get("in_relationship", ""),
                "Family_Relationship": request.POST.get("family_relationship", "0"),
                "Free_Time": request.POST.get("free_time", "0"),
                "Going_Out": request.POST.get("going_out", "0"),
                "Weekend_Alcohol_Consumption": request.POST.get("weekend_alcohol", "0"),
                "Weekday_Alcohol_Consumption": request.POST.get("weekday_alcohol", "0"),
                "Health_Status": request.POST.get("health_status", "0"),
                "Number_of_Absences": request.POST.get("absences", "0"),
                "Grade_1": request.POST.get("grade_1", "0"),
                "Grade_2": request.POST.get("grade_2", "0"),
                "Final_Grade": request.POST.get("final_grade", "0"),
            }

            # Validate and clean all fields
            student_data, warnings = DataValidator.validate_student_data(
                raw_student_data
            )
            validation_warnings.extend(warnings)

            print("Validation warnings:", warnings)
            for warning in warnings:
                print(f"  - {warning}")

            # Initialize unknown category handler
            category_handler = UnknownCategoryHandler(
                label_encoders, column_info.get("categorical", [])
            )

            # Build feature vector in the correct order
            X_input = []
            encoding_warnings = []

            for feature in feature_names:
                if feature in label_encoders:
                    # Categorical feature - use handler for unknown values
                    encoded_val, warning = category_handler.encode_categorical(
                        feature, student_data.get(feature, "unknown")
                    )
                    X_input.append(encoded_val)
                    if warning:
                        encoding_warnings.append(warning)
                        print(f"  {warning}")
                else:
                    # Numerical feature
                    try:
                        val = float(student_data.get(feature, 0))
                        X_input.append(val)
                    except (ValueError, TypeError) as e:
                        print(
                            f"Warning: Could not convert {feature}={student_data.get(feature, 0)} to float, using 0"
                        )
                        X_input.append(0)

            # Convert to numpy array
            X_input = np.array(X_input, dtype=float).reshape(1, -1)

            # Scale numerical features
            try:
                numerical_cols = column_info.get("numerical", [])
                if numerical_cols:
                    numerical_indices = [
                        feature_names.index(col)
                        for col in numerical_cols
                        if col in feature_names
                    ]
                    if numerical_indices:
                        X_numerical_only = X_input[:, numerical_indices]
                        X_numerical_scaled = scaler.transform(X_numerical_only)
                        X_input[:, numerical_indices] = X_numerical_scaled
                        print(
                            f"Successfully scaled {len(numerical_indices)} numerical features"
                        )
                    else:
                        print("No numerical indices found to scale")
                else:
                    print("No numerical columns in column_info")
            except Exception as scale_error:
                print(f"Scaling error: {str(scale_error)}")
                print(f"Numerical columns: {column_info.get('numerical', [])}")
                print(f"Feature names count: {len(feature_names)}")
                messages.error(request, f"Error scaling features: {str(scale_error)}")
                return render(request, "predict.html")

            # Make prediction
            prediction = model.predict(X_input)[0]
            probability = model.predict_proba(X_input)[0]
            dropout_prob = probability[1] * 100

            prediction_result = {
                "predicted_dropout": bool(prediction),
                "dropout_probability_percentage": round(dropout_prob, 2),
                "safe_probability_percentage": round(probability[0] * 100, 2),
                "risk_level": "HIGH RISK" if dropout_prob > 50 else "LOW RISK",
                "student_data": student_data,
                "validation_warnings": validation_warnings + encoding_warnings,
            }

            # Save prediction to database if user is logged in
            if request.user.is_authenticated:
                DropoutPrediction.objects.create(
                    user=request.user,
                    student_name=student_data["Student_Name"],
                    school=student_data["School"],
                    gender=student_data["Gender"],
                    age=int(student_data["Age"]),
                    address=student_data["Address"],
                    family_size=student_data["Family_Size"],
                    parental_status=student_data["Parental_Status"],
                    mother_education=int(student_data["Mother_Education"]),
                    father_education=int(student_data["Father_Education"]),
                    mother_job=student_data["Mother_Job"],
                    father_job=student_data["Father_Job"],
                    grade_1=int(student_data["Grade_1"]),
                    grade_2=int(student_data["Grade_2"]),
                    final_grade=int(student_data["Final_Grade"]),
                    number_of_failures=int(student_data["Number_of_Failures"]),
                    number_of_absences=int(student_data["Number_of_Absences"]),
                    study_time=int(student_data["Study_Time"]),
                    travel_time=int(student_data["Travel_Time"]),
                    wants_higher_education=student_data["Wants_Higher_Education"],
                    internet_access=student_data["Internet_Access"],
                    in_relationship=student_data["In_Relationship"],
                    health_status=int(student_data["Health_Status"]),
                    weekend_alcohol_consumption=int(
                        student_data["Weekend_Alcohol_Consumption"]
                    ),
                    weekday_alcohol_consumption=int(
                        student_data["Weekday_Alcohol_Consumption"]
                    ),
                    free_time=int(student_data["Free_Time"]),
                    going_out=int(student_data["Going_Out"]),
                    family_relationship=int(student_data["Family_Relationship"]),
                    school_support=student_data["School_Support"],
                    family_support=student_data["Family_Support"],
                    dropout_risk_percentage=dropout_prob,
                    predicted_dropout=bool(prediction),
                )
                messages.success(request, "Prediction saved to your records!")

        except Exception as e:
            import traceback

            error_detail = traceback.format_exc()
            print(f"Error making prediction: {error_detail}")
            messages.error(request, f"Error making prediction: {str(e)}")

    context = {
        "prediction_result": prediction_result,
        "model_available": model is not None,
    }
    return render(request, "predict.html", context)


# View Prediction History
@login_required(login_url="login")
def prediction_history(request):
    """View all predictions made by the user"""
    predictions = DropoutPrediction.objects.filter(user=request.user).order_by(
        "-created_at"
    )

    context = {
        "predictions": predictions,
        "total_predictions": predictions.count(),
        "high_risk_count": predictions.filter(predicted_dropout=True).count(),
        "low_risk_count": predictions.filter(predicted_dropout=False).count(),
    }
    return render(request, "prediction_history.html", context)


# Edit User Profile
@login_required(login_url="login")
def edit_profile(request):
    """Edit user profile information"""
    user = request.user

    if request.method == "POST":
        # Get form data
        first_name = request.POST.get("first_name", "")
        last_name = request.POST.get("last_name", "")
        email = request.POST.get("email", "")

        # Validate email
        if email and email != user.email:
            if (
                User.objects.filter(email=email)
                .exclude(username=user.username)
                .exists()
            ):
                messages.error(request, "Email already in use by another account.")
                return render(request, "edit_profile.html", {"user": user})

        # Update user information
        user.first_name = first_name
        user.last_name = last_name
        user.email = email
        user.save()

        messages.success(request, "Profile updated successfully!")
        return redirect("dashboard")

    context = {"user": user, "page_title": "Edit Profile"}
    return render(request, "edit_profile.html", context)


# Edit Prediction
@login_required(login_url="login")
def edit_prediction(request, prediction_id):
    """Edit a saved prediction record"""
    try:
        prediction = DropoutPrediction.objects.get(id=prediction_id, user=request.user)
    except DropoutPrediction.DoesNotExist:
        messages.error(request, "Prediction not found.")
        return redirect("prediction_history")

    if request.method == "POST":
        # Update prediction fields
        prediction.student_name = request.POST.get(
            "student_name", prediction.student_name
        )
        prediction.school = request.POST.get("school", prediction.school)
        prediction.gender = request.POST.get("gender", prediction.gender)
        prediction.age = int(request.POST.get("age", prediction.age))
        prediction.address = request.POST.get("address", prediction.address)
        prediction.family_size = request.POST.get("family_size", prediction.family_size)
        prediction.parental_status = request.POST.get(
            "parental_status", prediction.parental_status
        )
        prediction.mother_education = int(
            request.POST.get("mother_education", prediction.mother_education)
        )
        prediction.father_education = int(
            request.POST.get("father_education", prediction.father_education)
        )
        prediction.mother_job = request.POST.get("mother_job", prediction.mother_job)
        prediction.father_job = request.POST.get("father_job", prediction.father_job)
        prediction.grade_1 = int(request.POST.get("grade_1", prediction.grade_1))
        prediction.grade_2 = int(request.POST.get("grade_2", prediction.grade_2))
        prediction.final_grade = int(
            request.POST.get("final_grade", prediction.final_grade)
        )
        prediction.number_of_failures = int(
            request.POST.get("failures", prediction.number_of_failures)
        )
        prediction.number_of_absences = int(
            request.POST.get("absences", prediction.number_of_absences)
        )
        prediction.study_time = int(
            request.POST.get("study_time", prediction.study_time)
        )
        prediction.travel_time = int(
            request.POST.get("travel_time", prediction.travel_time)
        )
        prediction.wants_higher_education = request.POST.get(
            "higher_education", prediction.wants_higher_education
        )
        prediction.internet_access = request.POST.get(
            "internet_access", prediction.internet_access
        )
        prediction.in_relationship = request.POST.get(
            "in_relationship", prediction.in_relationship
        )
        prediction.health_status = int(
            request.POST.get("health_status", prediction.health_status)
        )
        prediction.weekend_alcohol_consumption = int(
            request.POST.get("weekend_alcohol", prediction.weekend_alcohol_consumption)
        )
        prediction.weekday_alcohol_consumption = int(
            request.POST.get("weekday_alcohol", prediction.weekday_alcohol_consumption)
        )
        prediction.free_time = int(request.POST.get("free_time", prediction.free_time))
        prediction.going_out = int(request.POST.get("going_out", prediction.going_out))
        prediction.family_relationship = int(
            request.POST.get("family_relationship", prediction.family_relationship)
        )
        prediction.school_support = request.POST.get(
            "school_support", prediction.school_support
        )
        prediction.family_support = request.POST.get(
            "family_support", prediction.family_support
        )

        try:
            # Re-evaluate with ML Model
            model, label_encoders, scaler, feature_names, column_info = load_ml_model()
            if model and scaler and feature_names and column_info:
                # Prepare data dictionary matching predict_dropout_risk schema
                student_data = {
                    "Student_Name": prediction.student_name,
                    "School": prediction.school,
                    "Gender": prediction.gender,
                    "Age": prediction.age,
                    "Address": prediction.address,
                    "Family_Size": prediction.family_size,
                    "Parental_Status": prediction.parental_status,
                    "Mother_Education": prediction.mother_education,
                    "Father_Education": prediction.father_education,
                    "Mother_Job": prediction.mother_job,
                    "Father_Job": prediction.father_job,
                    "Reason_for_Choosing_School": getattr(
                        prediction, "reason_for_choosing_school", ""
                    ),
                    "Guardian": getattr(prediction, "guardian", ""),
                    "Travel_Time": prediction.travel_time,
                    "Study_Time": prediction.study_time,
                    "Number_of_Failures": prediction.number_of_failures,
                    "School_Support": prediction.school_support,
                    "Family_Support": prediction.family_support,
                    "Extra_Paid_Class": getattr(prediction, "extra_paid_class", ""),
                    "Extra_Curricular_Activities": getattr(
                        prediction, "extra_curricular_activities", ""
                    ),
                    "Attended_Nursery": getattr(prediction, "attended_nursery", ""),
                    "Wants_Higher_Education": prediction.wants_higher_education,
                    "Internet_Access": prediction.internet_access,
                    "In_Relationship": prediction.in_relationship,
                    "Family_Relationship": prediction.family_relationship,
                    "Free_Time": prediction.free_time,
                    "Going_Out": prediction.going_out,
                    "Weekend_Alcohol_Consumption": prediction.weekend_alcohol_consumption,
                    "Weekday_Alcohol_Consumption": prediction.weekday_alcohol_consumption,
                    "Health_Status": prediction.health_status,
                    "Number_of_Absences": prediction.number_of_absences,
                    "Grade_1": prediction.grade_1,
                    "Grade_2": prediction.grade_2,
                    "Final_Grade": prediction.final_grade,
                }

                # Transform to feature array
                X_input = []
                for feature in feature_names:
                    if feature in label_encoders:
                        le = label_encoders[feature]
                        try:
                            encoded_val = le.transform(
                                [str(student_data.get(feature, ""))]
                            )[0]
                            X_input.append(encoded_val)
                        except:
                            X_input.append(0)
                    else:
                        try:
                            val = float(student_data.get(feature, 0))
                            X_input.append(val)
                        except:
                            X_input.append(0)

                X_input = np.array(X_input, dtype=float).reshape(1, -1)

                numerical_indices = [
                    feature_names.index(col) for col in column_info["numerical"]
                ]
                X_numerical_only = X_input[:, numerical_indices]
                X_numerical_scaled = scaler.transform(X_numerical_only)
                X_input[:, numerical_indices] = X_numerical_scaled

                pred_val = model.predict(X_input)[0]
                probability = model.predict_proba(X_input)[0]

                prediction.dropout_risk_percentage = probability[1] * 100
                prediction.predicted_dropout = bool(pred_val)
        except Exception as e:
            print(f"Error recalculating prediction during edit: {str(e)}")

        prediction.save()
        messages.success(request, "Prediction updated successfully!")
        return redirect("prediction_history")

    context = {
        "prediction": prediction,
        "page_title": f"Edit Prediction - {prediction.student_name}",
    }
    return render(request, "edit_prediction.html", context)


@login_required(login_url="login")
def change_password(request):
    """View to change current user's password"""
    if request.method == "POST":
        form = PasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            user = form.save()
            update_session_auth_hash(request, user)  # Important!
            messages.success(request, "Your password was successfully updated!")
            return redirect("dashboard")
        else:
            messages.error(request, "Please correct the error below.")
    else:
        form = PasswordChangeForm(request.user)
    return render(request, "change_password.html", {"form": form})


def forgot_password(request):
    if request.method == "POST":
        email = request.POST.get("email")
        try:
            user = User.objects.get(email=email)
            # Generate 6-digit OTP
            otp = "".join([str(random.randint(0, 9)) for _ in range(6)])
            
            # Save OTP to database
            PasswordResetOTP.objects.filter(user=user).delete() # Delete old OTPs
            PasswordResetOTP.objects.create(user=user, otp=otp)
            
            # Send email
            subject = "Your Password Reset OTP"
            message = f"Hello {user.username},\n\nYour OTP for password reset is: {otp}\n\nThis OTP is valid for 10 minutes.\n\nIf you did not request this, please ignore this email."
            send_mail(subject, message, None, [email])
            
            request.session['reset_email'] = email
            messages.success(request, f"An OTP has been sent to {email}.")
            return redirect("verify_otp")
        except User.DoesNotExist:
            messages.error(request, "No account found with this email address.")
            
    return render(request, "forgot_password.html")


def verify_otp(request):
    email = request.session.get('reset_email')
    if not email:
        messages.error(request, "Session expired. Please try again.")
        return redirect("forgot_password")
        
    if request.method == "POST":
        otp_input = request.POST.get("otp")
        try:
            user = User.objects.get(email=email)
            otp_record = PasswordResetOTP.objects.get(user=user, otp=otp_input)
            
            # Check if OTP is older than 10 minutes
            if timezone.now() - otp_record.created_at > timedelta(minutes=10):
                otp_record.delete()
                messages.error(request, "OTP has expired. Please request a new one.")
                return redirect("forgot_password")
                
            otp_record.is_verified = True
            otp_record.save()
            
            request.session['otp_verified'] = True
            messages.success(request, "OTP verified successfully. You can now reset your password.")
            return redirect("reset_password")
        except PasswordResetOTP.DoesNotExist:
            messages.error(request, "Invalid OTP. Please try again.")
            
    return render(request, "verify_otp.html", {"email": email})


def reset_password(request):
    email = request.session.get('reset_email')
    otp_verified = request.session.get('otp_verified')
    
    if not email or not otp_verified:
        messages.error(request, "Session expired or OTP not verified.")
        return redirect("forgot_password")
        
    if request.method == "POST":
        password = request.POST.get("password")
        password2 = request.POST.get("password2")
        
        if password != password2:
            messages.error(request, "Passwords do not match.")
            return render(request, "reset_password.html")
            
        try:
            user = User.objects.get(email=email)
            user.set_password(password)
            user.save()
            
            # Clean up session and OTP
            PasswordResetOTP.objects.filter(user=user).delete()
            del request.session['reset_email']
            del request.session['otp_verified']
            
            messages.success(request, "Password reset successful. You can now log in with your new password.")
            return redirect("login")
        except Exception as e:
            messages.error(request, f"Error resetting password: {str(e)}")
            
    return render(request, "reset_password.html")


# ── Custom Error Handlers ──────────────────────────────────────────────────────

def error_400(request, exception=None):
    return render(request, "400.html", status=400)


def error_403(request, exception=None):
    return render(request, "403.html", status=403)


def error_404(request, exception=None):
    return render(request, "404.html", status=404)


def error_500(request):
    return render(request, "500.html", status=500)
