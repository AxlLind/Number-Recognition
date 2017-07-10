
#ifndef NUMBER_RECOGNITION_MNISTREADER_H
#define NUMBER_RECOGNITION_MNISTREADER_H

#include <ios>
#include <fstream>
#include <iostream>
#include "Matrix.h"
typedef unsigned char byte;

/**
 * Static class to read and parse data from the "The MNIST database of handwritten digits".
 * Data file path hardcoded as static member variables at the bottom.
 * Values returned as Matrix objects (see Matrix.h).
 * Documentation on how data is structured in the files: http://yann.lecun.com/exdb/mnist/
 *
 * Example:
 *
 * Matrix training_data = MNISTParser::Parse( MNISTParser::TrainingData );
 *
 * @author Axel Lindeberg
 * @date 2017-07-01
 */
class MNISTParser {
    static const std::string training_data_path, training_label_path, test_data_path, test_label_path;

    static int byteToInt(byte a, byte b, byte c, byte d) {
        return (a << 24) | (b << 16) | (c << 8) | d;
    }

    static std::vector<byte> read(std::string file_name) {
        std::ifstream file(file_name, std::ios::binary|std::ios::ate);
        auto pos = file.tellg();

        std::vector<char> data(pos);

        file.seekg(0, std::ios::beg);
        file.read(&data[0], pos);
        return std::vector<byte>(data.begin(), data.end());
    }

    static Matrix parseLabels(bool testing = false) {
        auto data = read(testing ? test_label_path : training_label_path);
        int items = 500;
        // int items = byteToInt(data[4], data[5], data[6], data[7]);

        Matrix OUT(items, 10);
        for (int i = 0; i < items; ++i)
            OUT(i, data[i+8], 1);
        return OUT;
    }

    static Matrix parseData(bool testing = false) {
        auto data = read(testing ? test_data_path : training_data_path);
        int items = 500;
        // int items = byteToInt(data[4],  data[5],  data[6],  data[7]);
        int rows  = byteToInt(data[8],  data[9],  data[10], data[11]);
        int cols  = byteToInt(data[12], data[13], data[14], data[15]);
        int pixels = rows * cols;

        Matrix OUT(items, pixels);
        for (int i = 0; i < items; ++i) {
            for (int j = 0; j < pixels; ++j) {
                double val = static_cast<double>(data[i * cols + j + 16]) / 255;
                OUT(i, j, val);
            }
        }
        return OUT;
    }

    MNISTParser() {/* To prevent instantiation */}

 public:
    enum DataType { TrainingData, TrainingLabels, TestData, TestLabels };

    static Matrix Parse(DataType type) {
        switch (type) {
            case TrainingData:   return parseData();
            case TrainingLabels: return parseLabels();
            case TestData:       return parseData(true);
            case TestLabels:     return parseLabels(true);

            default:
                throw std::invalid_argument("MNISTParser::Parse() - Invalid parameter (somehow?!)");
        }
    }
};


const std::string MNISTParser::training_data_path  = "../Data/TrainingData/train-images-idx3-ubyte";
const std::string MNISTParser::training_label_path = "../Data/TrainingData/train-labels-idx1-ubyte";
const std::string MNISTParser::test_data_path      = "../Data/TestData/t10k-images-idx3-ubyte";
const std::string MNISTParser::test_label_path     = "../Data/TestData/t10k-labels-idx1-ubyte";

#endif //NUMBER_RECOGNITION_MNISTREADER_H
