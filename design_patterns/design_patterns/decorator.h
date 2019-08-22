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

//! Derived class: concrete component
class HouseBlend : public Coffee
{
public:
    HouseBlend() { description_ += ": HouseBlend"; }

    ~HouseBlend() { std::cout << "~HouseBlend()" << std::endl; }

    std::string getDescription() override { return description_; }

    double cost() override { return 1.99; }
};

//! Derived class: concrete component
class DarkRoast : public Coffee
{
public:
    DarkRoast() { description_ += ": DarkRoast"; }

    ~DarkRoast() { std::cout << "~DarkRoast()" << std::endl; }

    std::string getDescription() override { return description_; }

    double cost() override { return 2.99; }
};

//! Decorator base is a derived class of the abstract component base
class Decorator : public Coffee
{
public:
    Decorator() { description_ = "Decorator";  }

    ~Decorator() { std::cout << "~Decorator()" << std::endl; }

	virtual std::string getDescription() = 0;
	
	virtual double cost() = 0;
};

inline void testDecorator()
{

}