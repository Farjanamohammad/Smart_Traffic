# import os
# import subprocess  # ✅ Used to call predictspeed.py
# from django.shortcuts import render, redirect
# from django.core.files.storage import FileSystemStorage
# from django.conf import settings
# from django.views import View
# from django.contrib.auth import authenticate, login, logout
# from .models import CustomUser, UploadedVideo  # ✅ Import models
# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt

# def process_uploaded_video(video_path):
#     command = ["python", "predictspeed.py", "--video", video_path]
#     subprocess.run(command, check=True)

# # ✅ Home page (Login Page)
# def home(request):
#     return render(request, 'app/login.html')

# # ✅ User Registration View
# def register(request):
#     if request.method == 'POST':
#         first_name = request.POST.get('first_name')
#         last_name = request.POST.get('last_name')
#         email = request.POST.get('email')
#         password = request.POST.get('password')
#         address = request.POST.get('address')
#         adharcard = request.POST.get('adharcard')
#         phone = request.POST.get('phone')
#         age = request.POST.get('age')

#         # ✅ Check if Aadhar card already exists (Avoid UNIQUE constraint error)
#         if CustomUser.objects.filter(adharcard=adharcard).exists():
#             return render(request, 'app/register.html', {'error': 'Aadhar card already registered!'})

#         # ✅ Create user with hashed password using Django Authentication
#         user = CustomUser.objects.create_user(
#             first_name=first_name,
#             last_name=last_name,
#             email=email,
#             password=password,  # ✅ `create_user()` automatically hashes password
#             address=address,
#             adharcard=adharcard,
#             phone=phone,
#             age=age
#         )
#         user.save()
#         return redirect('login')  # Redirect to login
#     return render(request, 'app/register.html')

# # ✅ Login View
# class LoginView(View):
#     def get(self, request):
#         return render(request, 'app/login.html')

#     def post(self, request, *args, **kwargs):
#         email = request.POST.get('email')
#         password = request.POST.get('password')

#         # ✅ Use `authenticate()` instead of manually checking the password
#         user = authenticate(request, email=email, password=password)

#         if user is not None:
#             login(request, user)  # ✅ Logs in the user properly
#             return redirect('dashboard')
#         else:
#             return render(request, 'app/login.html', {'error': 'Invalid credentials'})

# # ✅ Dashboard View
# def dashboard(request):
#     if not request.user.is_authenticated:  # ✅ Secure check
#         return redirect('login')

#     context = {'email': request.user.email}
#     return render(request, 'app/main.html', context)

# # ✅ Logout View
# def logout_view(request):
#     logout(request)  # ✅ Clears session safely
#     return redirect('login')

# # ✅ Video Upload View & Call predicts_speed.py
# @csrf_exempt  # ✅ Allows AJAX POST request
# def upload_video(request):
#     if request.method == 'POST':
#         print("📩 Received POST request")

#         if request.FILES:
#             print("✅ Uploaded files:", request.FILES)
#         else:
#             print("❌ No files uploaded!")

#         if request.FILES.get('video'):
#             video_file = request.FILES['video']
#             print("📂 Received file:", video_file.name)

#             # ✅ Save file to media/uploads/
#             upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
#             os.makedirs(upload_dir, exist_ok=True)  # ✅ Ensure the directory exists
#             fs = FileSystemStorage(location=upload_dir)
#             filename = fs.save(video_file.name, video_file)
#             video_path = fs.path(filename)

#             print("✅ Video uploaded successfully:", video_path)

#             # ✅ Call `predictspeed.py`
#             try:
#                 print("🚀 Running predictspeed.py on:", video_path)
#                 process_uploaded_video(video_path)
#                 return JsonResponse({'message': 'Processing started!', 'video_path': video_path})
#             except Exception as e:
#                 print("❌ Error running predictspeed.py:", e)
#                 return JsonResponse({'error': str(e)}, status=500)

#     return JsonResponse({'error': 'No video uploaded'}, status=400)


############################################################################################################################

# import os
# import subprocess  # ✅ Used to call predictspeed.py
# from django.shortcuts import render, redirect
# from django.core.files.storage import FileSystemStorage
# from django.conf import settings
# from django.views import View
# from django.contrib.auth import authenticate, login, logout
# from .models import CustomUser, UploadedVideo  # ✅ Import models
# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# # from app.models import SpeedViolation, HelmetViolation  # ✅ Ensure models are imported

# from app.models import SpeedViolation


# def process_uploaded_video(video_path):
#     command = ["python", "predictspeed.py", "--video", video_path]
#     subprocess.run(command, check=True)

# # ✅ Home page (Login Page)
# def home(request):
#     return render(request, 'app/login.html')

# # ✅ User Registration View
# def register(request):
#     if request.method == 'POST':
#         first_name = request.POST.get('first_name')
#         last_name = request.POST.get('last_name')
#         email = request.POST.get('email')
#         password = request.POST.get('password')
#         address = request.POST.get('address')
#         adharcard = request.POST.get('adharcard')
#         phone = request.POST.get('phone')
#         age = request.POST.get('age')

#         # ✅ Check if Aadhar card already exists (Avoid UNIQUE constraint error)
#         if CustomUser.objects.filter(adharcard=adharcard).exists():
#             return render(request, 'app/register.html', {'error': 'Aadhar card already registered!'})

#         # ✅ Create user with hashed password using Django Authentication
#         user = CustomUser.objects.create_user(
#             first_name=first_name,
#             last_name=last_name,
#             email=email,
#             password=password,  # ✅ `create_user()` automatically hashes password
#             address=address,
#             adharcard=adharcard,
#             phone=phone,
#             age=age
#         )
#         user.save()
#         return redirect('login')  # Redirect to login
#     return render(request, 'app/register.html')

# # ✅ Login View
# class LoginView(View):
#     def get(self, request):
#         return render(request, 'app/login.html')

#     def post(self, request, *args, **kwargs):
#         email = request.POST.get('email')
#         password = request.POST.get('password')

#         # ✅ Use `authenticate()` instead of manually checking the password
#         user = authenticate(request, email=email, password=password)

#         if user is not None:
#             login(request, user)  # ✅ Logs in the user properly
#             return redirect('dashboard')
#         else:
#             return render(request, 'app/login.html', {'error': 'Invalid credentials'})

# # ✅ Dashboard View
# def dashboard(request):
#     if not request.user.is_authenticated:  # ✅ Secure check
#         return redirect('login')

#     violations = SpeedViolation.objects.all().order_by('-timestamp')
#     context = {
#         'email': request.user.email,
#         'violations': violations
#     }
#     return render(request, 'app/main.html', context)  # ✅ Single return statement
# # ✅ Logout View
# def logout_view(request):
#     logout(request)  # ✅ Clears session safely
#     return redirect('login')

# # ✅ Video Upload View & Call predict_speed.py
# @csrf_exempt  # ✅ Allows AJAX POST request
# def upload_video(request):
#     if request.method == 'POST':
#         if request.FILES.get('video'):
#             video_file = request.FILES['video']
#             upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
#             os.makedirs(upload_dir, exist_ok=True)  # ✅ Ensure the directory exists
#             fs = FileSystemStorage(location=upload_dir)
#             filename = fs.save(video_file.name, video_file)
#             video_path = fs.path(filename)

#             try:
#                 process_uploaded_video(video_path)
#                 return JsonResponse({'message': 'Processing started!', 'video_path': video_path})
#             except Exception as e:
#                 return JsonResponse({'error': str(e)}, status=500)
#     return JsonResponse({'error': 'No video uploaded'}, status=400)



# # ✅ Speed Violation List View
# def speed_violation_list(request):
#     return render(request, 'app/speedviolations.html')
# def detect_speed(request):
#     """ ✅ Fetch only recent speed violations """
#     violations = SpeedViolation.objects.all().order_by('-timestamp')[:10]

#     data = {
#         "violations": [
#             {
#                 "plate_number": v.plate_number,
#                 "speed": v.speed,
#                 "image_url": v.image.url if v.image else ""
#             }
#             for v in violations
#         ]
#     }
#     return JsonResponse(data)



##################################################################################################

# latest one


import os
import subprocess  # ✅ Used to call external scripts
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.views import View
from django.contrib.auth import authenticate, login, logout
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from app.models import CustomUser, UploadedVideo, SpeedViolation, HelmetViolation  # ✅ Import models

# ✅ Run predictspeed.py asynchronously
def process_uploaded_video(video_path):
    command = ["python", "predictspeed.py", "--video", video_path]
    subprocess.Popen(command)  # Runs in the background

# ✅ Home page (Login Page)
def home(request):
    return render(request, 'app/login.html')

# ✅ User Registration View
def register(request):
    if request.method == 'POST':
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        email = request.POST.get('email')
        password = request.POST.get('password')
        address = request.POST.get('address')
        adharcard = request.POST.get('adharcard')
        phone = request.POST.get('phone')
        age = request.POST.get('age')

        if CustomUser.objects.filter(adharcard=adharcard).exists():
            return render(request, 'app/register.html', {'error': 'Aadhar card already registered!'})

        user = CustomUser.objects.create_user(
            first_name=first_name,
            last_name=last_name,
            email=email,
            password=password,
            address=address,
            adharcard=adharcard,
            phone=phone,
            age=age
        )
        user.save()
        return redirect('login')
    return render(request, 'app/register.html')

# ✅ Login View
class LoginView(View):
    def get(self, request):
        return render(request, 'app/login.html')

    def post(self, request, *args, **kwargs):
        email = request.POST.get('email')
        password = request.POST.get('password')

        user = authenticate(request, email=email, password=password)

        if user is not None:
            login(request, user)
            return redirect('dashboard')
        else:
            return render(request, 'app/login.html', {'error': 'Invalid credentials'})

# ✅ Dashboard View
def dashboard(request):
    if not request.user.is_authenticated:
        return redirect('login')

    speed_violations = SpeedViolation.objects.all().order_by('-timestamp')
    helmet_violations = HelmetViolation.objects.all().order_by('-timestamp')
    context = {
        'email': request.user.email,
        'speed_violations': speed_violations,
        'helmet_violations': helmet_violations
    }
    return render(request, 'app/main.html', context)

# ✅ Logout View
def logout_view(request):
    logout(request)
    return redirect('login')

# ✅ Video Upload View & Call predict_speed.py
@csrf_exempt
def upload_video(request):
    if request.method == 'POST':
        if request.FILES.get('video'):
            video_file = request.FILES['video']
            upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
            os.makedirs(upload_dir, exist_ok=True)
            fs = FileSystemStorage(location=upload_dir)
            filename = fs.save(video_file.name, video_file)
            video_path = fs.path(filename)

            try:
                process_uploaded_video(video_path)
                return JsonResponse({'message': 'Processing started!', 'video_path': video_path})
            except Exception as e:
                return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'No video uploaded'}, status=400)

# ✅ Speed Violation List View
def speed_violation_list(request):
    return render(request, 'app/speedviolations.html')

# ✅ API for Speed Violations
def detect_speed(request):
    violations = SpeedViolation.objects.all().order_by('-timestamp')[:10]
    data = {
        "violations": [
            {
                "plate_number": v.plate_number,
                "speed": v.speed,
                "image_url": v.image.url if v.image else ""
            }
            for v in violations
        ]
    }
    return JsonResponse(data)

# ✅ API for Helmet Violations
def detect_helmet_violations(request):
    violations = HelmetViolation.objects.all().order_by('-timestamp')[:10]
    data = {
        "helmet_violations": [  # ✅ Fixed key name
            {
                "plate_number": v.plate_number,
                "image_url": v.image.url if v.image else ""
            }
            for v in violations
        ]
    }
    return JsonResponse(data)

# ✅ Number Plate Extraction API
def extract_number_plate(request):
        # Fetch extracted plates from the database
        plates = SpeedViolation.objects.all().order_by('-timestamp')[:10]
        data = {
            "plates": [
                {
                    "plate_number": v.plate_number,
                    "image_url": v.image.url if v.image else ""
                }
                for v in plates
            ]
        }
        return JsonResponse(data)

# ✅ Helmet Detection API
def detect_helmet(request):
        # Fetch helmet violations from the database
        helmet_violations = HelmetViolation.objects.all().order_by('-timestamp')[:10]
        data = {
            "helmet_violations": [
                {
                    "plate_number": v.plate_number,
                    "helmet_status": "No Helmet" if not v.helmet_detected else "Helmet On",
                    "image_url": v.image.url if v.image else ""
                }
                for v in helmet_violations
            ]
        }
        return JsonResponse(data)

