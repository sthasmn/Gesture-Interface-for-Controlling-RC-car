import RPi.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

Handle = 18
Motor = 22

GPIO.setup(Handle, GPIO.OUT)
GPIO.setup(Motor, GPIO.OUT)
pwm_motor = GPIO.PWM(Motor, 50)
pwm_handle = GPIO.PWM(Handle, 50)

def angle_to_percent(angle):
    if angle > 180 or angle < 0:
        return False
    start = 4
    end = 10
    ratio = (end - start)/180

    angle_as_percent = angle * ratio
    return start + angle_as_percent

#motor stop at 5.8 to 6.3
#backward from 5.7 to 5.0 or less
#forward from 6.4 to 7.0 or more
def motor_speed(motor, handle_angle):
    pwm_motor.start(motor)
    handle(handle_angle)

def handle(handle_value):
    pwm_handle.start(angle_to_percent(handle_value))
    sleep(0.05)
'''
#test for handle and speed
for i in range(50, 70):
    i = float(i)
    motor_speed(i/10, 90)
    print(i)
    sleep(1)


    return False
start = 4
end = 10
ratio = (end - start)/180

angle_as_percent = angle * ratio
return start + angle_as_percent

#motor stop at 5.8 to 6.3
#backward from 5.7 to 5.0 or less
#forward from 6.4 to 7.0 or more
def motor_speed(motor, handle_angle):
    pwm_motor.start(motor)
    handle(handle_angle)

def handle(handle_value):
    pwm_handle.start(angle_to_percent(handle_value))
    sleep(0.05)
'''
#test for handle and speed
for i in range(50, 70):
    i = float(i)
    motor_speed(i/10, 90)
    print(i)
    sleep(1)
