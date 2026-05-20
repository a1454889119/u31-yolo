#include <assert.h>
#include <fstream>
#include <string>
#include "UpixLayerInfo.h"

#pragma warning(disable:4996)

typedef std::vector<std::string> StringArray;
static std::string trim(const std::string& text, const std::string& chars)
{
	std::string newText;
	size_t start = 0, length = text.length(), end = length - 1;
	for (size_t i = 0; i < length; i++)
	{
		int n = (int)chars.find_first_of(text[i]);
		if (n >= 0) {
			start++;
		}
		else
			break;
	}

	for (int i = (int)length - 1; i >= 0; i--) {
		int n = (int)chars.find_first_of(text[i]);
		if (n >= 0) {
			end--;
		}
		else
			break;
	}

	return text.substr(start, end - start + 1);
}

static StringArray splitByChars(const std::string& text, const std::string& chars)
{
	StringArray parts;
	std::string t;
	for (auto c : text) {
		int n = (int)chars.find_first_of(c);
		if (n >= 0) {
			parts.push_back(t);
			t.clear();
		}
		else {
			t += c;
		}
	}
	if (t.length() > 0) {
		parts.push_back(t);
	}
	return parts;
}

bool parseDecimal(const std::string& text, int* v)
{
	int len = (int)text.size();
	if (len == 0 || len > 10) return false;
	if (len > 1 && text[0] == ('0')) return false;
	uint64_t value = 0;
	for (int i = 0; i < len; i++) {
		int n = text[i] - ('0');
		if (n < 0 || n >= 10) return false;
		value += n;
		if (i != len - 1)
			value *= 10;
	}

	if (value > INT_MAX)
		return false;
	*v = (int)value;
	return true;
}

static std::vector<StringArray> loadCSV(const std::string filename)
{
	std::vector<StringArray> result;
	std::ifstream ifs(filename);

	while (!ifs.eof()) {
		std::string line;
		getline(ifs, line);
		auto text = trim(line, "\r\n\t\" ");
		auto parts = splitByChars(text, ",");
		StringArray trimed_parts;
		for (auto text : parts) {
			trimed_parts.push_back(trim(text, "\r\n\t\" "));
		}
		result.push_back(trimed_parts);
	}
	ifs.close();
	return result;
}

static const char* layer_type_names[(int)UpixLayerType::MAX] = {
	"conv",
	"maxp",
	"shuffle",
	"upsample",
	"mul",
	"add",
	"global_avg_pool",
	"reduce_max",
	"reduce_mean",
};


static const char* activate_names[(int)ActivateType::MAX] = {
	"linear",
	"leaky",
	"relu",
	"sigmoid",
};

static UpixLayerInfo decodeLine(const StringArray& line)
{
	UpixLayerInfo info;
	int i;
	if (line.size() != 10)
		throw std::string("Need 10 parts in a line");

	const std::string idx_name = line[0];
	const std::string io_name = line[1];
	const std::string type_name = line[2];
	const std::string act_name = line[3];
	const std::string ksize_name = line[4];
	const std::string stride_name = line[5];
	const std::string pad_name = line[6];
	const std::string groups_name = line[7];
	const std::string inputs_name = line[8];
	const std::string output_types = line[9];

	if (idx_name.size() != 5 || idx_name[0] != '[' || idx_name[4] != ']') {
		throw "bad idx :" + idx_name;
	}


	if (sscanf(idx_name.c_str(), "[%3d]", &info.idx) != 1)
		throw "bad idx :" + idx_name;

	if (6 != sscanf(io_name.c_str(), "%3d*%3d*%3d->%3d*%3d*%3d",
		&info.inw, &info.inh, &info.inc,
		&info.outw, &info.outh, &info.outc))
		throw "Wrong in->out '" + io_name + "'";

	for (i = 0; i < (int)UpixLayerType::MAX; i++) {
		if (type_name == layer_type_names[i]) {
			info.type = (UpixLayerType)i;
			break;
		}
	}
	if (i == (int)UpixLayerType::MAX) {
		throw "Unknown type name '" + type_name + "'";
	}

	if (!act_name.empty()) {
		for (i = 0; i < (int)ActivateType::MAX; i++) {
			if (act_name == activate_names[i]) {
				info.activate = (ActivateType)i;
				break;
			}
		}
		if (i == (int)ActivateType::MAX) {
			throw "Unknown activate name '" + act_name + "'";
		}
	}

	if (!ksize_name.empty()) {
		if (!parseDecimal(ksize_name, &info.ksize))
			throw "Wrong ksize '" + ksize_name + "'";
	}

	if (!stride_name.empty()) {
		if (!parseDecimal(stride_name, &info.stride))
			throw "Wrong stride '" + stride_name + "'";
	}

	if (!pad_name.empty()) {
		if (!parseDecimal(pad_name, &info.pad))
			throw "Wrong pad '" + pad_name + "'";
	}

	if (!groups_name.empty()) {
		if (!parseDecimal(groups_name, &info.groups))
			throw "Wrong pad '" + groups_name + "'";
	}
	if (!inputs_name.empty()) {
		auto parts = splitByChars(inputs_name, ";");
		for (auto& p : parts) {
			stInputSource from;
			if (3 != sscanf(p.c_str() + 3, "[%3d]%2d+%2d", &from.layer, &from.channel_start, &from.channel_count)) {
				throw "Bad content in 'inputs_name' : " + p;
			}
			info.isrcs.push_back(from);
		}
	}
	if (!output_types.empty()) {
		auto parts = splitByChars(output_types, " ");
		int out = (int)OutputType::None;
		for (auto& p : parts) {
			if (p == "whc") {
				out |= (int)OutputType::WHC;
			}
			else if (p == "cwh") {
				out |= (int)OutputType::CWH;
			}
			else if (p == "any") {
				out |= (int)OutputType::CWH;
			}
			else {
				throw "Bad content in 'output_types' : " + p;
			}
		}
		info.outType = (OutputType)out;
	}
	return info;
}

int UpixLayerInfo::load(const char* filename, std::vector<UpixLayerInfo>& layers)
{
	layers.clear();
	auto lines = loadCSV(filename);

	try
	{
		for (auto& line : lines) {
			if (line.size() <= 0) continue;
			if (line.size() == 1 && line[0].empty()) continue;

			if (line[0].size() >= 2 && line[0][0] == '/' && line[0][1] == '/') {
				continue;
			}

			layers.push_back(decodeLine(line));
		}
				
		for (auto& info : layers) {
			for (auto& from : info.isrcs) {
				layers[from.layer].refcount++;
			}
		}
	}
	catch (const std::string& err)
	{
		printf("%s\n", err.c_str());
		layers.clear();
	}

	return (int)layers.size();
}
