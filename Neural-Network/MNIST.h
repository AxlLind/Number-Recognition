
#ifndef NUMBER_RECOGNITION_MNISTREADER_H
#define NUMBER_RECOGNITION_MNISTREADER_H

#include <fstream>
#include <vector>
#include "Matrix.h"

/**
 * Static class to read and parse data from the "The MNIST database of handwritten digits".
 * Data file path hardcoded as string variables at the bottom.
 * Values returned as Matrix objects (see Matrix.h).
 * Documentation on how data is structured in the files: http://yann.lecun.com/exdb/mnist/
 *
 * Example:
 *
 *      Matrix training_data = MNIST::ParseAll( MNIST::TrainingData );                  // returns 60000x784 matrix
 *      std::vector<Matrix> training_sets = MNIST::Parse( MNIST::TrainingData, 20, 10 ); // returns 10 20x784 matrices
 *
 * @author Axel Lindeberg
 * @date 2017-07-01
 */
class MNIST {
    static const std::string training_data_path, training_label_path, test_data_path, test_label_path;

    static int byteToInt(const std::vector<unsigned char> &data, int i) {
        return  (data[i] << 24) | (data[i+1] << 16) | (data[i+2] << 8) | data[i+3];
    }

    static std::vector<unsigned char> read(const std::string &file_name) {
        std::ifstream file(file_name, std::ios::binary|std::ios::ate);
        if (!file.is_open())
            throw std::runtime_error("MNIST::Parse() - Unable to read file");

        auto pos = file.tellg();
        std::vector<char> data(pos);

        file.seekg(0, std::ios::beg);
        file.read(&data[0], pos);
        return std::vector<unsigned char>(data.begin(), data.end());
    }

    static std::vector<Matrix<double>> parseLabels(int batch_size, bool test = false, int num_batches = 1) {
        auto data = read(test ? test_label_path : training_label_path);
        int items = byteToInt(data, 4);
        if (batch_size * num_batches > items)
            throw std::invalid_argument("MNIST::Parse() - number of items requested larger than data set");

        std::vector<Matrix<double>> OUT(num_batches, Matrix<double>(batch_size,10));
        int index = 8;
        for (int i = 0; i < num_batches; ++i) {
            for (int j = 0; j < OUT[i].rows(); ++j) {
                OUT[i](j, data[index++]) =  1;
            }
        }
        return OUT;
    }

    static std::vector<Matrix<double>> parseData(int batch_size, bool test = false, int num_batches = 1) {
        auto data = read(test ? test_data_path : training_data_path);
        int items = byteToInt(data, 4);

        if (batch_size * num_batches > items)
            throw std::invalid_argument("MNIST::Parse() - number of items requested larger than data set");

        int rows = byteToInt(data, 8);
        int cols = byteToInt(data, 12);
        int pixels = rows * cols;

        std::vector<Matrix<double>> OUT(num_batches, Matrix<double>(batch_size, pixels));
        int index = 16;
        for (int i = 0; i < num_batches; ++i) {
            for (int j = 0; j < batch_size; ++j) {
                for (int k = 0; k < pixels; ++k) {
                    double val = static_cast<double>(data[index++]) / 255;
                    OUT[i](j, k) = val;
                }
            }
        }
        return OUT;
    }

    MNIST() {/* To prevent instantiation */}

 public:
    enum DataType { TrainingData, TrainingLabels, TestData, TestLabels };

    /**
     * Parses the whole subset of the data set, as specified by the DataType.
     * Returns the parsed data as a Matrix.
     */
    static Matrix<double> ParseAll(DataType type) {
        switch (type) {
            case TrainingData:   return   parseData(60000)[0];
            case TrainingLabels: return parseLabels(60000)[0];
            case TestData:       return   parseData(10000, true)[0];
            case TestLabels:     return parseLabels(10000, true)[0];

            default:
                throw std::invalid_argument("MNIST::ParseAll() - Invalid DataType (somehow?!)");
        }
    }

    /**
     * Parses a subset of the data into different matrices and returns a vector containing them.
     * If the batch_size * num_batches exceeds the number of data points it will throw an error.
     *
     * @param batch_size How many datapoints each batch contains
     * @param num_batches How many batches to return
     */
    static std::vector<Matrix<double>> Parse(DataType type, int batch_size, int num_batches) {
        if (batch_size < 1 || num_batches < 1)
            throw std::invalid_argument("MNIST::Parse() - Batch size and/or number of batches need to be at least one");

        switch (type) {
            case TrainingData:   return   parseData(batch_size, false, num_batches);
            case TrainingLabels: return parseLabels(batch_size, false, num_batches);
            case TestData:       return   parseData(batch_size, true,  num_batches);
            case TestLabels:     return parseLabels(batch_size, true,  num_batches);

            default:
                throw std::invalid_argument("MNIST::Parse() - Invalid DataType (somehow?!)");
        }
    }
};

const std::string MNIST::training_data_path  = "../Data/train-images-idx3-ubyte",
                  MNIST::training_label_path = "../Data/train-labels-idx1-ubyte",
                  MNIST::test_data_path      = "../Data/t10k-images-idx3-ubyte" ,
                  MNIST::test_label_path     = "../Data/t10k-labels-idx1-ubyte";

#endif //NUMBER_RECOGNITION_MNIST_H
