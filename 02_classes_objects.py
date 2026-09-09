# self == this, in java sau javascript.
# all methods surrounded by __, like __str__ are Dunder methods.

class User:
    def __init__(self, name, age):
        self.name = name
        self.age = age
        self.nationality = "Romanian"

    def say_hello(self):
        print(self.name + " says hi!")

    def __str__(self):
        return "" + self.name + " " + str(self.age) + ", " + self.nationality

# folosind User() intializam o instanta a clasei User.
sonia = User("Sonia", 30)
dragos = User("Dragos", 35)

print(sonia)
print(dragos)

print(sonia.name)
sonia.say_hello()











