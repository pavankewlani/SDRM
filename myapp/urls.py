from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("login/", views.login_view, name="login"),
    path("register/", views.register_view, name="register"),
    path("logout/", views.logout_view, name="logout"),
    path("about/", views.about, name="about"),
    path("contact/", views.contact, name="contact"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("profile/edit/", views.edit_profile, name="edit_profile"),
    path("predict/", views.predict_dropout_risk, name="predict"),
    path("prediction-history/", views.prediction_history, name="prediction_history"),
    path(
        "prediction/<int:prediction_id>/edit/",
        views.edit_prediction,
        name="edit_prediction",
    ),
    path("password/change/", views.change_password, name="change_password"),
    path("password/forgot/", views.forgot_password, name="forgot_password"),
    path("password/verify-otp/", views.verify_otp, name="verify_otp"),
    path("password/reset/", views.reset_password, name="reset_password"),
]
