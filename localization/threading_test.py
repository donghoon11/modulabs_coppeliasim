import logging
import threading
import time

class ClassA:
    def run(self):
        while True:
            print('a')

class ClassB:
    def run(self):
        while True:
            print('b')

a = ClassA()
b = ClassB()

thread_a = threading.Thread(target=a.run)
thread_b = threading.Thread(target=b.run)

thread_a.start()
thread_b.start()

thread_a.join()
thread_b.join()
