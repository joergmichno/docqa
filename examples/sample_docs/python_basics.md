# Python Basics

Python is a high-level, interpreted programming language known for its clear syntax and readability. It was created by Guido van Rossum and first released in 1991. Python supports multiple programming paradigms, including procedural, object-oriented, and functional programming.

## Variables and Data Types

In Python, variables are created by assigning a value with the equals sign. Python uses dynamic typing, which means you do not need to declare the type of a variable before using it. The interpreter determines the type at runtime.

Common built-in data types include integers, floating-point numbers, strings, booleans, lists, tuples, dictionaries, and sets. Strings can be defined with single quotes, double quotes, or triple quotes for multi-line text. Lists are ordered and mutable collections, while tuples are ordered but immutable.

## Functions

Functions are reusable blocks of code defined with the `def` keyword. They accept parameters and can return values using the `return` statement. Python supports default parameter values, keyword arguments, and variable-length argument lists with `*args` and `**kwargs`.

A simple function looks like this: you write `def greet(name):` followed by an indented body that contains the logic. Functions help organise code into small, testable, and reusable units. They also improve readability by giving meaningful names to blocks of logic.

## Classes and Object-Oriented Programming

Python supports object-oriented programming through classes. A class is a blueprint for creating objects that bundle data (attributes) and behaviour (methods) together. You define a class with the `class` keyword and initialise instances using the `__init__` method.

Inheritance allows a class to reuse code from a parent class. Python supports single and multiple inheritance. Polymorphism lets different classes implement the same interface in different ways. Encapsulation is achieved through naming conventions, such as prefixing private attributes with an underscore.

## Modules and Packages

Python code is organised into modules and packages. A module is a single Python file, while a package is a directory containing an `__init__.py` file and one or more modules. The standard library provides hundreds of built-in modules for tasks like file handling, networking, and data processing.

You import modules with the `import` statement. You can import specific names with `from module import name` or give a module an alias with `import module as alias`. This system keeps code modular, avoids name collisions, and makes large projects manageable.

## Error Handling

Python uses try-except blocks for error handling. When an error occurs, Python raises an exception. You can catch specific exceptions with `except ExceptionType` or catch all exceptions with a bare `except` clause. The `finally` block runs regardless of whether an exception occurred, making it useful for cleanup tasks like closing files.
