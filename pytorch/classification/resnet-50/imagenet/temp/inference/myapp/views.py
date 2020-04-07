from django.shortcuts import render
from django.http import JsonResponse,HttpResponse
from django.views.decorators.clickjacking import xframe_options_exempt
from django.views.decorators.csrf import csrf_exempt
import requests,os
from myapp.utils.object_detection import processor_util
import tensorflow as tf


@xframe_options_exempt
def Userui(request):
    program = os.getenv('PROGRAM', '')
    if program == 'mnist':
        return render(request, 'mnist.html')
    elif program == 'catsdogs':
        return render(request,'catsanddogs.html')
    elif program == 'cifar':
        return render(request,'cifar.html')
    elif program == 'objdetect':
        return render(request,'objdetect.html')
    elif program == 'bolts':
        return render(request, 'bolts.html')

@xframe_options_exempt
def Webapp(request):
    return render(request, "webapp.html")

@xframe_options_exempt
@csrf_exempt
def Predict(request):
    if request.method == 'POST':
        serving_url = ""
        headers = {}
        if 'serving_url' in request.POST:
            serving_url = request.POST['serving_url']
            model_name = serving_url.split("/")[-1].split(":")[0]
            host = '%s.default.example.com' % model_name
            headers.update({'Host': host})
        else:
            serving_url = os.getenv('TF_SERVING_URL', '')
        result = requests.post(str(serving_url),data=request.POST['SIGNATURE'], headers=headers)
        if request.POST['program'] != 'objdetect':
            #print("inside")
            response=HttpResponse(result.text)
            response["Access-Control-Allow-Origin"] = "*"
            return response
        label_map = request.FILES['text_file'].read()
        data = request.POST['SIGNATURE']
        num_classes = request.POST['numofclasses']
        imagebase64=processor_util.process_output(data, label_map, result.text, int(num_classes))
        base64_string=imagebase64.decode("utf-8")
        response=JsonResponse({"result": base64_string})
        response["Access-Control-Allow-Origin"] = "*"
        return response
    response=JsonResponse({"error": "expecting POST request"})
    response["Access-Control-Allow-Origin"] = "*"
    return response
