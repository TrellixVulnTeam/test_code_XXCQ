/*

The Decorator Pattern:
- Open-Close principle: Classes should be open for extension but closed for modification.
- Decorator attaches additional responsibilities to an object dynamically.
- The Decorator Pattern provides an alternative to subclassing for extending behavior.
*/

#pragma once

#include <iostream>
#include <string>
#include <memory>

//! Abstract component class as base
class Coffee
{
public:
    Coffee() { description_ = "Coffee"; }

    virtual ~Coffee() { std::cout << "~Coffee()" << std::endl; }

    virtual std::string getDescription() { return description_; }

    virtual double cost() = 0;

protected:
    std::string description_;
};

//! Derived class: concrete component from base
class HouseBlend : public Coffee
{
public:
    HouseBlend() { description_ += " - HouseBlend"; }

    ~HouseBlend() { std::cout << "~HouseBlend()" << std::endl; }

    std::string getDescription() override { return description_; }

    double cost() override { return 1.99; }
};

//! Derived class: concrete component from base
class DarkRoast : public Coffee
{
public:
    DarkRoast() { description_ += " - DarkRoast"; }

    ~DarkRoast() { std::cout << "~DarkRoast()" << std::endl; }

    std::string getDescription() override { return description_; }

    double cost() override { return 2.99; }
};

//--------------------------------------------------------------------------------------
//! Decorator base is a derived class of the abstract component base
class Decorator : public Coffee
{
public:
    Decorator() {}

    ~Decorator() { std::cout << "~Decorator()" << std::endl; }

    virtual std::string getDescription() = 0;

    virtual double cost() = 0;
};

//! Derived concrete class from Decorator
class Milk : public Decorator
{
public:
    Milk(std::unique_ptr<Coffee>& ptr) : coffee_ptr_(std::move(ptr)) {}

    ~Milk() { std::cout << "~Milk()" << std::endl; }

    std::string getDescription() { return coffee_ptr_->getDescription() + ", Decorator - Milk"; }

    double cost() { return coffee_ptr_->cost() + 0.59; }

private:
    std::unique_ptr<Coffee> coffee_ptr_;
};

//! Derived concrete class from Decorator
class Mocha : public Decorator
{
public:
    Mocha(std::unique_ptr<Coffee>& ptr) : coffee_ptr_(std::move(ptr)) {}

    ~Mocha() { std::cout << "~Mocha()" << std::endl; }

    std::string getDescription() { return coffee_ptr_->getDescription() + ", Decorator - Mocha"; }

    double cost() { return coffee_ptr_->cost() + 1.20; }

private:
    std::unique_ptr<Coffee> coffee_ptr_;
};
