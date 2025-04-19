# from django.db import models
# from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin

# class CustomUserManager(BaseUserManager):
#     def create_user(self, email, password=None, **extra_fields):
#         if not email:
#             raise ValueError("The Email field must be set")
#         email = self.normalize_email(email)
#         extra_fields.setdefault('is_active', True)
#         user = self.model(email=email, **extra_fields)
#         user.set_password(password)  # ✅ Hash password
#         user.save(using=self._db)
#         return user

#     def create_superuser(self, email, password=None, **extra_fields):
#         extra_fields.setdefault('is_staff', True)
#         extra_fields.setdefault('is_superuser', True)

#         if extra_fields.get('is_staff') is not True:
#             raise ValueError("Superuser must have is_staff=True.")
#         if extra_fields.get('is_superuser') is not True:
#             raise ValueError("Superuser must have is_superuser=True.")

#         return self.create_user(email, password, **extra_fields)

# class CustomUser(AbstractBaseUser, PermissionsMixin):
#     first_name = models.CharField(max_length=100)
#     last_name = models.CharField(max_length=100)
#     email = models.EmailField(unique=True)
#     address = models.CharField(max_length=255)
#     adharcard = models.CharField(max_length=12, unique=True)
#     age = models.PositiveIntegerField()
#     phone = models.CharField(max_length=15)
#     is_active = models.BooleanField(default=True)
#     is_staff = models.BooleanField(default=False)

#     objects = CustomUserManager()

#     USERNAME_FIELD = 'email'
#     REQUIRED_FIELDS = ['first_name', 'last_name']

#     class Meta:
#         db_table = 'CustomUser'

#     def __str__(self):
#         return self.email
# class UploadedVideo(models.Model):
#     video = models.FileField(upload_to='uploads/')  # ✅ Saves in 'media/uploads/'
#     uploaded_at = models.DateTimeField(auto_now_add=True)

#     def __str__(self):
#         return f"Video: {self.video.name} uploaded at {self.uploaded_at}"


##############################################################################################################################


from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin

class CustomUserManager(BaseUserManager):
    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError("The Email field must be set")
        email = self.normalize_email(email)
        extra_fields.setdefault('is_active', True)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)  # ✅ Hash password
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)

        if extra_fields.get('is_staff') is not True:
            raise ValueError("Superuser must have is_staff=True.")
        if extra_fields.get('is_superuser') is not True:
            raise ValueError("Superuser must have is_superuser=True.")

        return self.create_user(email, password, **extra_fields)

class CustomUser(AbstractBaseUser, PermissionsMixin):
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    address = models.CharField(max_length=255)
    adharcard = models.CharField(max_length=12, unique=True)
    age = models.PositiveIntegerField()
    phone = models.CharField(max_length=15)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)

    objects = CustomUserManager()

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['first_name', 'last_name']

    class Meta:
        db_table = 'custom_user'  # ✅ Use lowercase table name for consistency

    def __str__(self):
        return self.email

# ✅ Uploaded Video Model
class UploadedVideo(models.Model):
    video = models.FileField(upload_to='uploads/')  # ✅ Saves in 'media/uploads/'
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Video: {self.video.name} uploaded at {self.uploaded_at}"

# ✅ Speed Violation Model

class SpeedViolation(models.Model):
    plate_number = models.CharField(max_length=20)
    speed = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)
    image = models.ImageField(upload_to='violations/', null=True, blank=True)  # ✅ Fix: Add image field
    violation_type = models.CharField(max_length=50, choices=[('Speed', 'Speed Violation'), ('Helmet', 'Helmet Violation')], default='Speed')  # ✅ Add this line

    def __str__(self):
        return f"{self.plate_number} - {self.speed} km/h"
        
class HelmetViolation(models.Model):
    plate_number = models.CharField(max_length=20)
    timestamp = models.DateTimeField(auto_now_add=True)
    image = models.ImageField(upload_to="helmet_violations/", null=True, blank=True)
    helmet_detected = models.BooleanField(default=False)
    vehicle_type = models.CharField(max_length=20, null=True, blank=True) 
    violation_type = models.CharField(max_length=50, choices=[('Speed', 'Speed Violation'), ('Helmet', 'Helmet Violation')], default='Helmet')  

    def __str__(self):
        return f"Helmet Violation - {self.plate_number} ({'Helmet On' if self.helmet_detected else 'No Helmet'})"


# ✅ Model for Recognized License Plate Records
class LicensePlateRecord(models.Model):
    plate_number = models.CharField(max_length=20, unique=True)
    image = models.ImageField(upload_to='license_plates/', blank=True, null=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.plate_number
