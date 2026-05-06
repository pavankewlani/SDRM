from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from django.db.models import Q
from .models import DropoutPrediction, ContactMessage


# Customize admin site
admin.site.site_header = "Student Dropout Prediction Admin"
admin.site.site_title = "Admin Portal"
admin.site.index_title = "Dashboard"


# DropoutPrediction Admin
@admin.register(DropoutPrediction)
class DropoutPredictionAdmin(admin.ModelAdmin):
    list_display = [
        "created_dt_display",
        "student_name_display",
        "school_display",
        "gender_display",
        "age_display",
        "final_grade_display",
        "absences_display",
        "risk_status_display",
        "risk_percentage_display",
    ]
    list_filter = [
        "predicted_dropout",
        "school",
        "user",
        "created_at",
    ]
    search_fields = ["student_name", "user__username", "school"]
    readonly_fields = ["created_at", "dropout_risk_percentage", "formatted_created_at"]
    date_hierarchy = "created_at"
    list_per_page = 25
    ordering = ["-created_at"]

    fieldsets = (
        (
            "Prediction Result",
            {
                "fields": (
                    "user",
                    "predicted_dropout",
                    "dropout_risk_percentage",
                    "created_at",
                    "formatted_created_at",
                )
            },
        ),
        (
            "Student Information",
            {
                "fields": (
                    "student_name",
                    "school",
                    "gender",
                    "age",
                    "address",
                    "family_size",
                    "parental_status",
                )
            },
        ),
        (
            "Education Information",
            {
                "fields": (
                    "mother_education",
                    "father_education",
                    "mother_job",
                    "father_job",
                )
            },
        ),
        (
            "Academic Performance",
            {
                "fields": (
                    "grade_1",
                    "grade_2",
                    "final_grade",
                    "number_of_failures",
                    "number_of_absences",
                )
            },
        ),
        (
            "Student Characteristics",
            {
                "fields": (
                    "study_time",
                    "travel_time",
                    "wants_higher_education",
                    "internet_access",
                    "in_relationship",
                )
            },
        ),
        (
            "Health & Lifestyle",
            {
                "fields": (
                    "health_status",
                    "weekend_alcohol_consumption",
                    "weekday_alcohol_consumption",
                    "free_time",
                    "going_out",
                    "family_relationship",
                )
            },
        ),
        (
            "Support",
            {"fields": ("school_support", "family_support")},
        ),
    )

    def created_dt_display(self, obj):
        from django.utils.timezone import localtime
        local_time = localtime(obj.created_at)
        return format_html('<span style="color: #64748b;">{}</span>', local_time.strftime("%b %d, %Y"))
    created_dt_display.short_description = "Date"
    
    def student_name_display(self, obj):
        return format_html('<strong>{}</strong>', obj.student_name)
    student_name_display.short_description = "Name"

    def school_display(self, obj):
        return obj.school
    school_display.short_description = "Sch"

    def gender_display(self, obj):
        return obj.gender
    gender_display.short_description = "Gen"

    def age_display(self, obj):
        return obj.age
    age_display.short_description = "Age"

    def final_grade_display(self, obj):
        return format_html('<strong>{}</strong>', obj.final_grade)
    final_grade_display.short_description = "Fin Grd"

    def absences_display(self, obj):
        return obj.number_of_absences
    absences_display.short_description = "Abs"

    def risk_status_display(self, obj):
        """Display risk status with color coding resembling the app badges"""
        if obj.predicted_dropout:
            return format_html(
                '<span style="background-color: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); padding: 4px 10px; border-radius: 999px; font-weight: bold; font-size: 11px; text-transform: uppercase;">High Risk</span>'
            )
        else:
            return format_html(
                '<span style="background-color: rgba(34, 197, 94, 0.15); color: #22c55e; border: 1px solid rgba(34, 197, 94, 0.3); padding: 4px 10px; border-radius: 999px; font-weight: bold; font-size: 11px; text-transform: uppercase;">Low Risk</span>'
            )
    risk_status_display.short_description = "Status"

    def risk_percentage_display(self, obj):
        """Display risk percentage identical to app progress bar"""
        percentage = round(obj.dropout_risk_percentage, 2)
        return format_html(
            '<div style="display: flex; align-items: center; gap: 8px; min-width: 140px;">'
            '<div style="flex: 1; height: 6px; background: rgba(0,0,0,0.1); border-radius: 3px; overflow: hidden;">'
            '<div style="height: 100%; width: {}%; background: linear-gradient(90deg, #6366f1 0%, #ef4444 100%); border-radius: 3px;"></div>'
            '</div>'
            '<span style="font-size: 0.8rem; font-weight: 600; color: #64748b;">{}%</span>'
            '</div>',
            percentage,
            percentage,
        )
    risk_percentage_display.short_description = "Risk %"

    def created_at_display(self, obj):
        """Display creation date in a readable format"""
        from django.utils.timezone import localtime

        local_time = localtime(obj.created_at)
        return local_time.strftime("%d-%m-%Y %H:%M")

    created_at_display.short_description = "Predicted On"

    def formatted_created_at(self, obj):
        """Formatted creation date for detail view"""
        from django.utils.timezone import localtime

        local_time = localtime(obj.created_at)
        return local_time.strftime("%A, %d %B %Y at %I:%M %p")

    formatted_created_at.short_description = "Created At (Formatted)"


# ContactMessage Admin
@admin.register(ContactMessage)
class ContactMessageAdmin(admin.ModelAdmin):
    list_display = [
        "subject_display",
        "name",
        "email",
        "read_status_display",
        "created_at_display",
    ]
    list_filter = ["is_read", "created_at"]
    search_fields = ["name", "email", "subject", "message"]
    readonly_fields = ["created_at", "formatted_created_at"]
    date_hierarchy = "created_at"
    list_per_page = 20
    ordering = ["-created_at"]

    fieldsets = (
        (
            "Message Status",
            {"fields": ("is_read", "created_at", "formatted_created_at")},
        ),
        (
            "Sender Information",
            {"fields": ("name", "email")},
        ),
        (
            "Message",
            {"fields": ("subject", "message")},
        ),
    )

    actions = ["mark_as_read", "mark_as_unread"]

    def subject_display(self, obj):
        """Display subject with truncation"""
        subject = obj.subject[:50]
        if len(obj.subject) > 50:
            subject += "..."
        return subject

    subject_display.short_description = "Subject"

    def read_status_display(self, obj):
        """Display read status with icons and colors"""
        if obj.is_read:
            return format_html(
                '<span style="background-color: #95a5a6; color: white; padding: 5px 10px; border-radius: 3px; font-weight: bold;">✓ Read</span>'
            )
        else:
            return format_html(
                '<span style="background-color: #3498db; color: white; padding: 5px 10px; border-radius: 3px; font-weight: bold;">⚡ Unread</span>'
            )

    read_status_display.short_description = "Status"

    def created_at_display(self, obj):
        """Display creation date in a readable format"""
        from django.utils.timezone import localtime

        local_time = localtime(obj.created_at)
        return local_time.strftime("%d-%m-%Y %H:%M")

    created_at_display.short_description = "Received On"

    def formatted_created_at(self, obj):
        """Formatted creation date for detail view"""
        from django.utils.timezone import localtime

        local_time = localtime(obj.created_at)
        return local_time.strftime("%A, %d %B %Y at %I:%M %p")

    formatted_created_at.short_description = "Received At (Formatted)"

    def mark_as_read(self, request, queryset):
        updated = queryset.update(is_read=True)
        self.message_user(request, f"✓ {updated} message(s) marked as read.")

    mark_as_read.short_description = "✓ Mark selected messages as read"

    def mark_as_unread(self, request, queryset):
        updated = queryset.update(is_read=False)
        self.message_user(request, f"⚡ {updated} message(s) marked as unread.")

    mark_as_unread.short_description = "⚡ Mark selected messages as unread"
