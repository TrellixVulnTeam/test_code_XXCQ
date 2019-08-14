#include <boost/uuid/uuid.hpp>             // uuid class
#include <boost/uuid/uuid_generators.hpp>  // generators
#include <boost/uuid/uuid_io.hpp>          // streaming operators etc.
#include <iostream>
#include <string>

//! Convert boost UUID to a string without dashes
std::string uuidToStringNoDashes(const boost::uuids::uuid& uuid)
{
    std::string uuid_str = boost::uuids::to_string(uuid);
    uuid_str.erase(std::remove(uuid_str.begin(), uuid_str.end(), '-'), uuid_str.end());
    return uuid_str;
}

//! String to UUID
boost::uuids::uuid stringToUuid(const std::string& uuid_str)
{
    boost::uuids::string_generator gen;
    return gen(uuid_str);
}

int main()
{
    std::vector<boost::uuids::uuid> vec_uuids;
    int num = 10, test_idx = 4;
    boost::uuids::uuid test_uuid;
    for (int i = 0; i < 10; ++i)
    {
        boost::uuids::uuid uuid = boost::uuids::random_generator()();
        if (i == test_idx)
        {
            test_uuid = uuid;
        }
        vec_uuids.push_back(uuid);
    }
    std::cout << test_uuid << std::endl;

    std::cout << "---------------" << std::endl << std::endl;

    for (auto& uuid : vec_uuids)
    {
        std::cout << uuid << std::endl;
    }

    vec_uuids.erase(std::remove(vec_uuids.begin(), vec_uuids.end(), test_uuid), vec_uuids.end());
    std::cout << "---------------" << std::endl << std::endl;
    for (auto& uuid : vec_uuids)
    {
        std::cout << uuid << std::endl;
    }

    return 0;
}
