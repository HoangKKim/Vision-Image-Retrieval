#include "query_function.h"

vector<Image> Database::getData() {
    return this->data;
}

Mat Database::getCentroid() {
    return this->centers;
}

vector<pair<string, int>>Database::getLabelsInfo() {
    return this->labels_info;
}


void Database::loadDatabase(string file_name, string database_type, string feature_type) {
    FileStorage fs(file_name, FileStorage::READ);

    if (!fs.isOpened()) {
        cout << "Failed to open file for reading." << endl;
        return;
    }

    FileNode nodes = fs[database_type];
    if (nodes.empty()) {
        return;
    }
    for (FileNodeIterator it = nodes.begin(); it != nodes.end(); ++it) {
        Image Image;
        Mat feature;
        Image.setFileName((string)(*it)["filename"]);
        Image.setLabel((string)(*it)["label"]);

        if (feature_type == "SIFT") {
            (*it)["sift"] >> feature;
            Image.setSIFT(feature);
        }
        else if (feature_type == "ORB") {
            (*it)["orb"] >> feature;
            Image.setORB(feature);
        }
        else if (feature_type == "Histogram") {
            (*it)["hist"] >> feature;
            Image.setHistogram(feature);
        }
        else if (feature_type == "Correlogram") {
            (*it)["correlogram"] >> feature;
            Image.setCorrelogram(feature);
        }
        else if (feature_type == "Combine") {
            (*it)["combine"] >> feature;
            Image.setCombine(feature);
        }
        this->data.push_back(Image);
    }
    fs.release();

}

void Database::loadCentroids(string file_name) {
    ifstream file(file_name, ios::binary);

    Mat data;

    if (file.is_open()) {
        //cout << "OpenFile";
        while (file.peek() != EOF) {

            int rows, cols, type;

            file.read((char*)&rows, sizeof(int));
            file.read((char*)&cols, sizeof(int));
            file.read((char*)&type, sizeof(int));

            data = Mat(rows, cols, type);
            file.read((char*)data.data, data.total() * data.elemSize());

        }
        file.close();
    }
    this->centers = data;
}

void Database::loadLabelsInfo(string file_name) {
    vector<pair<string, int>> labels;
    ifstream file(file_name);

    if (!file.is_open()) {
        cout << "Cannot open file" << endl;
        return;
    }

    string line;
    while (getline(file, line)) {
        istringstream iss(line);
        int count;
        string name;

        // number of a label in dataset
        iss >> count;
        getline(iss, name); // read the remaning of line

        // erase the white space  
        if (!name.empty() && name[0] == ' ') {
            name.erase(0, 1);
        }

        labels.push_back({ name, count });
    }
    this->labels_info = labels;
    file.close();
};