#ifndef CONFIG_H
#define CONFIG_H

#include "common_include.h"

using namespace std;

namespace buildmodel
{
class Config
{
private:
//    static std::shared_ptr<Config> _config;
    cv::FileStorage _file;
    static Config* mConfig;
    string mFilename;
    // set a new config file


    // mzm add,set a default file
//    void setParameterFile();
    
    Config();
 // private constructor makes a singleton
public:
    ~Config();  // close the file when deconstructing
    static Config* getInstancePtr();

    void setParameterFile( const std::string& filename );

    // access the parameter values
    template< typename T >
    T get( const std::string& key )
    {
        return T( mConfig->_file[key] );
    }
};
}

#endif // CONFIG_H
