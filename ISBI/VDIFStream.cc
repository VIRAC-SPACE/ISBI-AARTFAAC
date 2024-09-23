#include <iostream>
#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
#include <cstdint>

#include <Common/Stream/Descriptor.h>
#include "Common/Stream/FileStream.h"

#include "VDIFStream.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <fstream>


using namespace std;

VDIFStream::VDIFStream(string input_file) {
    samples_per_frame = 2000;
    number_of_headers = 1;
    number_of_frames = 0;    
    input_file_ = input_file;
    
    bool read_vdif = readVDIFHeader(input_file, header, 0);

    if (read_vdif) {
        print_vdif_header();
        data_frame_size = static_cast<int>(header.dataframe_length) * 8 - 32 +  16 * static_cast<int>(header.legacy_mode);
        current_timestamp = header.sec_from_epoch;

    }else {
        std::cerr << "Error reading VDIF header." << std::endl;
    }

}

    size_t VDIFStream::tryWrite(const void *ptr, size_t size){ return 0;}
    size_t VDIFStream::tryRead(void *ptr, size_t size){ return 0; }


string VDIFStream::get_input_file(){ return   input_file_;  }


bool VDIFStream::readVDIFHeader(const std::string filePath, VDIFHeader& header, int flag) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return false;
    }
    
    file.seekg(flag, std::ios::beg);
    file.read(reinterpret_cast<char*>(&header), sizeof(VDIFHeader));
    

    if (!file) {
        std::cerr << "Failed to read header from file: " << filePath << std::endl;
        return false;
    }
    
    file.close();
    return true;
}


bool VDIFStream::readVDIFData(const std::string filePath, uint32_t frame[][1][16], size_t samples_per_frame,  int flag) {
   // cout<< "flag " << flag << endl;

    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return false;
    }
    
  
    file.seekg(flag, std::ios::beg);
    size_t dataSize = samples_per_frame * 1 * 16 * sizeof(uint32_t);
    file.read(reinterpret_cast<char*>(frame), dataSize);

    if (!file) {
        std::cerr << "Failed to read data from file: " << filePath << std::endl;
        return false;
    }
    
    file.close();

    for (size_t i = 0; i < samples_per_frame; ++i) {
	uint64_t sample_time = header.sec_from_epoch * 1000000000 + 1;
    	sample_time_stamps[i] = sample_time;
    }

    return true;
}


void VDIFStream::printFirstRow(uint32_t (*frame)[1][16], size_t samples_per_frame) {
    if (samples_per_frame > 0) {
        // Allocate memory for the float32 array
        float (*float_frame)[1][16] = new float[samples_per_frame][1][16];

        // Convert values from uint32_t to float32
        for (size_t i = 0; i < samples_per_frame; ++i) {
            for (size_t j = 0; j < 16; ++j) {
                float_frame[i][0][j] = static_cast<float>(frame[i][0][j]);
            }
        }

        // Print the first row of the float32 array
        std::cout << "First row of the float32 array:" << std::endl;
        for (size_t j = 0; j < 16; ++j) {
            std::cout << float_frame[0][0][j] << " ";
        }
        std::cout << std::endl;

        // Clean up the allocated memory
        delete[] float_frame;
    }
}


void VDIFStream::read(void *ptr, size_t size){
    VDIFHeader header_for_frame;    
        
    readVDIFHeader(get_input_file(), header_for_frame, (sizeof(VDIFHeader)+ data_frame_size) * number_of_frames);
    

    readVDIFData(get_input_file(), reinterpret_cast<uint32_t (*)[1][16]>(ptr),  samples_per_frame, sizeof(VDIFHeader) * (number_of_headers) + data_frame_size * number_of_frames);
   
    //printFirstRow(frame, samples_per_frame);
    //exit(0);

    current_timestamp = header_for_frame.sec_from_epoch;
    
    number_of_headers +=1;
    number_of_frames +=1;

}


int VDIFStream::get_data_frame_size(){
    return  data_frame_size;	
}


uint8_t VDIFStream::get_log2_nchan(){
    return header.log2_nchan;
}

int VDIFStream::get_samples_per_frame(){
    return samples_per_frame;
}


uint32_t VDIFStream::get_current_timestamp(){
    return current_timestamp;
}


void VDIFStream::print_vdif_header() {
        std::cout << "VDIF Header read successfully:" << std::endl;
        std::cout << "sec_from_epoch: " << header.sec_from_epoch << std::endl;
        std::cout << "legacy_mode: " << static_cast<int>(header.legacy_mode) << std::endl;
        std::cout << "invalid: " << static_cast<int>(header.invalid) << std::endl;
        std::cout << "dataframe_in_second: " << header.dataframe_in_second << std::endl;
        std::cout << "ref_epoch: " << static_cast<int>(header.ref_epoch) << std::endl;
        std::cout << "dataframe_length: " << header.dataframe_length << std::endl;
        std::cout << "log2_nchan: " << static_cast<int>(header.log2_nchan) << std::endl;
        std::cout << "version: " << static_cast<int>(header.version) << std::endl;
        std::cout << "station_id: " << header.station_id << std::endl;
        std::cout << "thread_id: " << header.thread_id << std::endl;
        std::cout << "bits_per_sample: " << static_cast<int>(header.bits_per_sample) << std::endl;
        std::cout << "data_type: " << static_cast<int>(header.data_type) << std::endl;
        std::cout << "user_data1: " << header.user_data1 << std::endl;
        std::cout << "edv: " << static_cast<int>(header.edv) << std::endl;
        std::cout << "user_data2: " << header.user_data2 << std::endl;
        std::cout << "user_data3: " << header.user_data3 << std::endl;
        std::cout << "user_data4: " << header.user_data4 << std::endl;
}


void VDIFStream::print_vdif_header(VDIFHeader header_) {
        std::cout << "VDIF Header read successfully:" << std::endl;
        std::cout << "sec_from_epoch: " << header_.sec_from_epoch << std::endl;
        std::cout << "legacy_mode: " << static_cast<int>(header_.legacy_mode) << std::endl;
        std::cout << "invalid: " << static_cast<int>(header_.invalid) << std::endl;
        std::cout << "dataframe_in_second: " << header_.dataframe_in_second << std::endl;
        std::cout << "ref_epoch: " << static_cast<int>(header_.ref_epoch) << std::endl;
        std::cout << "dataframe_length: " << header_.dataframe_length << std::endl;
        std::cout << "log2_nchan: " << static_cast<int>(header_.log2_nchan) << std::endl;
        std::cout << "version: " << static_cast<int>(header_.version) << std::endl;
        std::cout << "station_id: " << header_.station_id << std::endl;
        std::cout << "thread_id: " << header_.thread_id << std::endl;
        std::cout << "bits_per_sample: " << static_cast<int>(header_.bits_per_sample) << std::endl;
        std::cout << "data_type: " << static_cast<int>(header.data_type) << std::endl;
        std::cout << "user_data1: " << header_.user_data1 << std::endl;
        std::cout << "edv: " << static_cast<int>(header.edv) << std::endl;
        std::cout << "user_data2: " << header_.user_data2 << std::endl;
        std::cout << "user_data3: " << header_.user_data3 << std::endl;
        std::cout << "user_data4: " << header_.user_data4 << std::endl;
}


VDIFStream::~VDIFStream() {}


