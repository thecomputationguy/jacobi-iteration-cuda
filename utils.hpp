template <typename T>
class hostCUDAVariable
{
    private:
        /* data */
        T* x_ ;
        T* xd_ ;
        const size_t size_;
        const bool useGPU_;
    
    public:       
        hostCUDAVariable(const size_t size, const bool useGPU);
        void copyToDevice();
        void copyToHost();
        T*& getDeviceVariable();
        T*& getHostVariable();
        ~hostCUDAVariable();
};