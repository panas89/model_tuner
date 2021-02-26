import pickle

def dumpObjects(file,filename):
    """Method to dump objects
    https://pythontips.com/2013/08/02/what-is-pickle-in-python/
    """
    fileObject = open(filename,'wb')
    # this writes the object a to the
    # file named 'testfile'
    pickle.dump(file,fileObject)

    # here we close the fileObject
    fileObject.close()
    print('Object saved!')
    return;

def loadObjects(filename):
    """Method to load objects"""
    # we open the file for reading
    fileObject = open(filename,'rb')
    # load the object from the file into var b
    print('Object loaded!')
    return pickle.load(fileObject)
