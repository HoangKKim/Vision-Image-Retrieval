#include "train_function.h"

vector<Image>Dataset::getData() {
    return this->data;
}

void Dataset::setData(vector<Image>data) {
    this->data = data;
}

Mat Dataset::getCentroids() {
    return this->centroids;
}
void Dataset::setCentroids(Mat centers) {
    this->centroids = centers;
}

vector<Image> Dataset::readCSV(const string& folder, const string filename_csv) {
    string filename = folder + "/" + filename_csv;
    ifstream file(filename);

    vector<Image> data;
    string line;
    if (!file.is_open()) {
        cout << "Cannot open file";
    }

    bool flag = false;
    while (getline(file, line)) {
        if (flag == false) {
            flag = true;
            continue;
        }

        stringstream lineStream(line);
        string image_filename, image_label;
        Image Image;

        getline(lineStream, image_filename, ',');
        getline(lineStream, image_label, ',');
        while ((image_filename).length() < 5) {
            image_filename = "0" + image_filename;
        }
        image_filename = folder + "/images/" + image_filename + ".png";

        Image.setFileName(image_filename);
        Image.setLabel(image_label);

        data.push_back(Image);
    }
    this->data = data;
    return data;
}

string Dataset::getFileName(const string& path) {
    // find the position of last ('/') or ('\\') in path of file
    size_t pos = path.find_last_of("/\\");

    // return all path of file if do not find the position
    if (pos == string::npos) {
        return path;
    }

    // return the name of file
    return path.substr(pos + 1);
}

// create label for CD dataset
vector<Image> Dataset::createLabel(const string& folder) {
    // read all images in training folder
    vector<Image> data;

    vector <string> fileName;
    glob(folder, fileName, false);
    for (size_t i = 0; i < fileName.size(); i++) {
        Image Image;
        string image_filename, image_label;

        // get the file name and label of each image
        string name = getFileName(fileName[i]);
        image_filename = fileName[i];
        image_label = name.substr(0, 2);

        Image.setFileName(image_filename);
        Image.setLabel(image_label);
        
        // add these infor into data
        data.push_back(Image);
    }
    this->data = data;
    return data;
}

void Dataset::writeXML(vector <Image> data, string filenName_XML, string type) {

    FileStorage fs(filenName_XML, FileStorage::WRITE);
    if (!fs.isOpened()) {
        cout << "Failed to open file for writing." << endl;
        return;
    }
    fs << type << "{";
    for (size_t i = 0; i < data.size(); ++i) {
        fs << "item" << "{";
        fs << "filename" << data[i].getFileName();
        fs << "label" << data[i].getLabel();
        fs << "sift" << data[i].getSIFT();
        fs << "orb" << data[i].getORB();
        fs << "hist" << data[i].getHistogram();
        fs << "correlogram" << data[i].getCorrelogram();
        fs << "combine" << data[i].getCombine();
        fs << "}";
    }
    fs << "}";
    fs.release();

}

vector<Image> Dataset::readXML(string fileName_XML, string type) {
    vector <Image> data;

    FileStorage fs(fileName_XML, FileStorage::READ);

    if (!fs.isOpened()) {
        cout << "Failed to open file for reading." << endl;
        return{};
    }

    FileNode nodes = fs[type];
    if (nodes.empty()) {
        return{};
    }
    for (FileNodeIterator it = nodes.begin(); it != nodes.end(); ++it) {
        Image Image;
        Mat sift, orb, histogram, correlogram, combine;
        Image.setFileName((string)(*it)["filename"]);
        Image.setLabel((string)(*it)["label"]);
        (*it)["sift"] >> sift;
        (*it)["orb"] >> orb;
        (*it)["hist"] >> histogram;
        (*it)["correlogram"] >> correlogram;
        (*it)["combine"] >> combine;

        Image.setSIFT(sift);
        Image.setORB(orb);
        Image.setHistogram(histogram);
        Image.setCorrelogram(correlogram);
        Image.setCombine(combine);

        data.push_back(Image);
    }
    fs.release();
    this->data = data;
    return data;
}


void Dataset::save_BinaryFile(Mat descriptor, string fileName) {
    ofstream fo(fileName, ios::binary);

    if (fo.is_open()) {
        // features
        int rows = descriptor.rows;
        int cols = descriptor.cols;
        int type = descriptor.type();

        fo.write((char*)&rows, sizeof(int));
        fo.write((char*)&cols, sizeof(int));
        fo.write((char*)&type, sizeof(int));
        fo.write((char*)descriptor.data, descriptor.total() * descriptor.elemSize());

        fo.close();
    }
    else {
        cout << "Unable to open file for writing: " << fileName << endl;
    }
}

void Dataset::save_BinaryFile(vector<Mat> descriptors, string fileName) {
    ofstream fo(fileName, ios::binary);

    if (fo.is_open()) {
        for (int i = 0; i < descriptors.size(); i++) {
            Mat descriptor = descriptors[i];
            
            // features
            int rows = descriptor.rows;
            int cols = descriptor.cols;
            int type = descriptor.type();

            fo.write((char*)&rows, sizeof(int));
            fo.write((char*)&cols, sizeof(int));
            fo.write((char*)&type, sizeof(int));
            fo.write((char*)descriptor.data, descriptor.total() * descriptor.elemSize());
        }
        

        fo.close();
    }
    else {
        cout << "Unable to open file for writing: " << fileName << endl;
    }
}


Mat Dataset::load_BinaryFile(const string& filename) {
    ifstream file(filename, ios::binary);

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
        cout << "Load database successfully";
    }
    return data;
}

vector<Mat> Dataset::load_BinaryFile_vector(const string& filename) {
    ifstream file(filename, ios::binary);
    vector<Mat> colorHist;
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

            colorHist.push_back(data);
        }
        file.close();
        cout << "Load database successfully";
    }
    return colorHist;
}

// write txt file
void Dataset::countLabels(string data_filename, string data_type, string save_file) {
    Dataset data;
    data.readXML(data_filename, data_type);
    // count label
    vector<pair<string, int>> LabelsInfo;
    int count = 1;
    string previousLabel = data.getData()[0].getLabel();
    for (int i = 1; i < data.getData().size(); i++) {
        string currentLabel = data.getData()[i].getLabel();
        if (currentLabel != previousLabel) {
            LabelsInfo.push_back(make_pair(previousLabel, count));
            count = 1;
            previousLabel = currentLabel;
        }
        else {
            count++;
        }
    }
    // save file
    ofstream file(save_file);

    // Cannot open file
    if (!file.is_open()) {
        cout << "Cannot open file" << endl;
        return;
    }

    // write label for each image
    for (const auto& pair : LabelsInfo) {
        file << pair.second << " " << pair.first << endl;
    }
    //close file
    file.close();
}