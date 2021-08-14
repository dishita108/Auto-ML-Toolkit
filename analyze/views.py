from django.shortcuts import render
import sys
from subprocess import run, PIPE
from django.core.files.storage import FileSystemStorage
import pandas as pd
import os
# Create your views here.

def index(request):
    return render(request, "index.html")

def output(request):
    '''cv = request.POST.get('cv')
    iters = request.POST.get('iters')'''
    cv = '3'
    iters = '5'
    request_file = request.FILES['document'] if 'document' in request.FILES else None
    corr = request.POST.get('corr')
    task = request.POST.get('task')
    print("Task is :    ",task)

    print(type(request_file))
    print("file is", request_file)
    fs = FileSystemStorage()
    filename = fs.save(request_file.name, request_file)
    print("type filename : ",type(filename))
    
    data = pd.read_csv(os.path.join("media/",filename))
    data_f = data.head(n = 10)
    data.describe(include='all').to_csv('static/data_description.csv')
    #data_html = data_f.to_html(classes="table table-hover text-left text-white  position-relative" ,justify="left")
    #des_html = data_f1.to_html(classes="table table-hover text-left text-white position-relative",justify="left")
    fileurl = fs.open(filename)
    print("type fileurl : ",type(fileurl))
    templateurl = fs.url(filename)
    print("file full url", fileurl)
    print("template url",templateurl)
    print("filename",filename)
    out1 = run([sys.executable, 'model.py',cv,iters,corr,str(fileurl),task], shell=False, stdout=PIPE)
    #request_file= run([sys.executable,'models.py',str(fileurl),filename],shell=False, stdout=PIPE )
    data =out1.stdout.decode("utf-8").splitlines()
    
    if task == 'classification':
       data1 = data[1:7] 
       data2 = data[10:16]
       data3 = data[17:23]
    elif task == 'regression':
       data1 = data[1:6] 
       data2 = data[9:14]
       data3 = data[15:20]        
    '''data1 = data[1:9]
    data2 = data[11:20]
    data3 = data[21:29]'''
    temp = zip(data1, data2, data3)
    data4 = data[-8]
    data5 = data[-7]
    data6 = data[-5]
    data7 = data[-4:-2]
    
    print("Result from out1 : ",out1.stdout)
    c = data[-2:]
    #print(request_file.stdout)
    return render(request, "result.html", {'temp':temp, 'data4':data4, 'data5':data5, 'data6':data6, 'data7':data7,'c':c})

def aboutus(request):
    return render(request,"aboutus.html")