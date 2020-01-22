#include "decorator.h"

int main()
{
    {
        // First customer: wants a Dark Roast coffee only, no decorations
        std::unique_ptr<Coffee> darkroast_ptr = std::make_unique<DarkRoast>();
        std::cout << darkroast_ptr->getDescription() << std::endl;
        std::cout << "Cost: " << darkroast_ptr->cost() << std::endl;
    }

    {
        // Second customer: wants House Blend and add decorations later
        std::cout << "------------------------------------" << std::endl;
        std::unique_ptr<Coffee> houseblend_ptr = std::make_unique<HouseBlend>();
        std::cout << houseblend_ptr->getDescription() << std::endl;
        std::cout << "Cost: " << houseblend_ptr->cost() << std::endl;
		
		// Add milk
		std::unique_ptr<Coffee> milk_ptr = std::make_unique<Milk>(houseblend_ptr);
		if (!houseblend_ptr)
		{
            std::cout << "Pointer houseblend_ptr is cleared now." << std::endl;
		}
        std::cout << milk_ptr->getDescription() << std::endl;
        std::cout << "Cost: " << milk_ptr->cost() << std::endl;

		// Add mocha
        std::unique_ptr<Coffee> mocha_ptr = std::make_unique<Mocha>(milk_ptr);
        if (!milk_ptr)
        {
            std::cout << "Pointer milk_ptr is cleared now." << std::endl;
        }
        std::cout << mocha_ptr->getDescription() << std::endl;
        std::cout << "Cost: " << mocha_ptr->cost() << std::endl;
    }

    return 0;
}