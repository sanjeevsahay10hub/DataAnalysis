import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE','studentproject.settings')
import django
django.setup()

from testapp.models import Students
from faker import Faker
from random import *

fake=Faker()

def phonenumbergen():
    d1=randint(6,9)#this willchoose any number from 6 to 9
    num= str(d1) # this will convert d1 int value in string
    for i in range(9):
        num=num+str(randint(0,9))
    return int(num)

def populate(n):
    for i in range(n):
        frollno=fake.random_int(min=1,max=999)
        fname=fake.name()
        fdob=fake.date()
        fmarks=fake.random_int(min=1, max=100)
        femail=fake.email()
        fphonenumber=phonenumbergen()
        faddress=fake.address()
        student_record=Students.objects.get_or_create(rollno=frollno, name=fname,
        dob=fdob,marks=fmarks,email=femail,phonenumber=fphonenumber, address=faddress)

populate(30)
