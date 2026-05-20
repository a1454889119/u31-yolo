#include <string>
#include <vector>
#include <fstream>

static std::vector<std::string> load_output_names(const char* filename)
{
	std::vector<std::string> results;
	std::ifstream ifs(filename);

	while (!ifs.eof()) {
		std::string line;
		getline(ifs, line);
		results.push_back(line);
	}
	ifs.close();
	return results;
}