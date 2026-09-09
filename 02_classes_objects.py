# self == this, in java sau javascript.

class User:
    def __init__(self, name):
        self.name = name

    def say_hello(self):
        print(self.name + " says hi!")


sonia = User("Sonia")
dragos = User("Dragos")

print(sonia)
print(dragos)

print(sonia.name)










