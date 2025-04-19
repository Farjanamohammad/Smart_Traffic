# from django.contrib import admin
# from .models import CustomUser  # Import your User model
# @admin.register(CustomUser)
# class UserAdmin(admin.ModelAdmin):
#     list_display = ('email', 'first_name', 'last_name', 'address', 'phone', 'age','password')

from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import CustomUser

@admin.register(CustomUser)
class UserAdmin(BaseUserAdmin):
    list_display = ('email', 'first_name', 'last_name', 'address', 'phone', 'age', 'is_staff')
    search_fields = ('email', 'first_name', 'last_name', 'phone')
    ordering = ('email',)
    list_filter = ('is_staff', 'is_active')

    fieldsets = (
        ("User Info", {"fields": ("first_name", "last_name", "email", "password")}),
        ("Personal Details", {"fields": ("address", "phone", "age", "adharcard")}),
        ("Permissions", {"fields": ("is_staff", "is_active", "is_superuser", "groups", "user_permissions")}),
    )

    add_fieldsets = (
        (None, {
            "classes": ("wide",),
            "fields": ("email", "first_name", "last_name", "password1", "password2", "is_staff", "is_active"),
        }),
    )
