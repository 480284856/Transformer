import requests
from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.
import json
from django.views.decorators.http import require_http_methods

# @require_http_methods(['POST'])
def index(request):

	msg = {}
	if request.method == "GET":
		return render(request=request,template_name='index.html')
	else:
		data1 = request.POST['inputs']
		if data1 != '':
			msg['orgn_seq'] = data1
			data = data1.encode('utf-8').decode('latin1')
			res = requests.post("http://localhost:6666/predictions/translator", data=data.encode('utf-8').decode('latin1'))  # 传到模型里面去并给出答案
			res.encoding = 'utf-8'
			msg['rlt_fra'] = res.text
			result = render(request=request,template_name='translate.html',context=msg)
			request = None
			return result
		else:
			return render(request=request,template_name='index.html')