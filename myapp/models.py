from django.db import models
from django.contrib.auth.models import User


# Student Dropout Prediction Model
class DropoutPrediction(models.Model):
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="dropout_predictions",
        null=True,
        blank=True,
    )

    # Student Information
    student_name = models.CharField(max_length=100, default="")
    school = models.CharField(max_length=10)
    gender = models.CharField(max_length=10)
    age = models.IntegerField()
    address = models.CharField(max_length=10)
    family_size = models.CharField(max_length=10)
    parental_status = models.CharField(max_length=20)

    # Education Information
    mother_education = models.IntegerField()
    father_education = models.IntegerField()
    mother_job = models.CharField(max_length=50)
    father_job = models.CharField(max_length=50)

    # Academic Performance
    grade_1 = models.IntegerField()
    grade_2 = models.IntegerField()
    final_grade = models.IntegerField()
    number_of_failures = models.IntegerField()
    number_of_absences = models.IntegerField()

    # Student Characteristics
    study_time = models.IntegerField()
    travel_time = models.IntegerField()
    wants_higher_education = models.CharField(max_length=10)
    internet_access = models.CharField(max_length=10)
    in_relationship = models.CharField(max_length=10)

    # Health & Lifestyle
    health_status = models.IntegerField()
    weekend_alcohol_consumption = models.IntegerField()
    weekday_alcohol_consumption = models.IntegerField()
    free_time = models.IntegerField()
    going_out = models.IntegerField()
    family_relationship = models.IntegerField()

    # Support
    school_support = models.CharField(max_length=10)
    family_support = models.CharField(max_length=10)

    # Prediction Result
    dropout_risk_percentage = models.FloatField(default=0.0)
    predicted_dropout = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        risk_level = "High" if self.predicted_dropout else "Low"
        return f"{self.school} Student - {risk_level} Risk ({self.dropout_risk_percentage:.1f}%)"


# Contact Message Model
class ContactMessage(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()
    subject = models.CharField(max_length=200)
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.subject} - from {self.name}"


class PasswordResetOTP(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    otp = models.CharField(max_length=6)
    created_at = models.DateTimeField(auto_now_add=True)
    is_verified = models.BooleanField(default=False)

    def __str__(self):
        return f"OTP for {self.user.email}"
