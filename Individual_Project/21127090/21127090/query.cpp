#include "query_function.h"

vector<pair<string, string>> Query::getImages() {
	return this->images;
}

void Query::loadQueryImages(string file_name) {
	ifstream file(file_name);
	if (!file.is_open()) {
		cout << "Cannot open the file" << endl;
		return;
	}

	// read the first line is the path file 
	string imagePath;
	getline(file, imagePath);
	cout << "Image Path: " << imagePath << endl;

	string line;
	// read the next lines include file name and its label
	while (getline(file, line)) {
		istringstream iss(line);
		string filename, label;
		iss >> filename;
		getline(iss, label);

		// remove whitespace
		if (!label.empty() && label[0] == ' ') {
			label.erase(0, 1);
		}

		this->images.push_back(make_pair(imagePath + filename, label));
	}

	file.close();

}