from osgeo import gdal, osr
import numpy as np


# function return metadata of gdal supported file
# meta data includes
def getMeta(filename):
    metadata = {}
    # Open the file:
    dataset = gdal.Open(filename, gdal.GA_ReadOnly)
    
    # Check type of the variable 'dataset'
    type(dataset)
    
   # Dimensions and # Number of bands    
    metadata['nbands'] = dataset.RasterCount 
    metadata['width'] = dataset.RasterXSize
    metadata['height'] = dataset.RasterYSize

    # Read dataset  properties:
    metadata['type'] = dataset.GetDriver().ShortName
    metadata['type_fullname'] = dataset.GetDriver().LongName


    # Projection
    metadata['projection'] = dataset.GetProjection()
    # projection information
    geotransform = dataset.GetGeoTransform()
    metadata['geotransform'] = geotransform

    # Metadata for the raster dataset
    metadata['metadata'] = dataset.GetMetadata()

    band = dataset.GetRasterBand(1)
    metadata['datatype'] = gdal.GetDataTypeName(band.DataType)
    metadata['overviews'] = band.GetOverviewCount()
    metadata['nodata'] = band.GetNoDataValue()
    dataset = None
    return metadata


#read array from input file
def getData(filename,bands = None, xoff = 0, yoff = 0, xsize= 0,ysize = 0):

    # Open the file:
    dataset = gdal.Open(filename, gdal.GA_ReadOnly)
    
    if bands == None:
        # data = np.
        bands = np.array(range(dataset.RasterCount)) + 1
   
    # data = np. 
    for i,bandi in enumerate(bands):
        band = dataset.GetRasterBand(bandi)    #get current band
        #transfer band into numpy array (row will become first e.g. band[cols,rows] will be array [rows, cols]) 
        arr = band.ReadAsArray(xoff, yoff, xsize, ysize) 
        #arr = band.ReadAsArray(startcolumn, startrow, column, row) 
        if i == 0 :
            data = arr
        else:
            #data = np.concatenate((data,img), axis = 0) 
            data = np.dstack((data,arr))  #Stacking Along Height (depth)
    dataset = None
    return data


#read array from input file
def getAllData(filename):
    
    # Open the file:
    dataset = gdal.Open(filename, gdal.GA_ReadOnly)
  
    # data = np. 
    nbands = dataset.RasterCount 
    for i  in range(nbands):
        band = dataset.GetRasterBand(i + 1)    #get current band
        #transfer band into numpy array (row will become first e.g. band[cols,rows] will be array [rows, cols]) 
        arr = band.ReadAsArray() 
        #arr = band.ReadAsArray(startcolumn, startrow, column, row) 
        if i == 0 :
            data = arr
        else:
            #data = np.concatenate((data,img), axis = 0) 
            data = np.dstack((data,arr))  #Stacking Along Height (depth)
    dataset = None
    return data


#save array to an gdal supported file e.g. TIFF, HDF5
def creatFile(output_file, out_data, 
              XSize = 0, YSize = 0, nbands = 0,
              filetype = "GTiff", datatype = 2,
              outbands = None, novalue = None, 
              geo_transform = '',projection = ''):
# Create gtif file
    #driver = gdal.GetDriverByName("GTiff")
    driver = gdal.GetDriverByName(filetype)
    
    if XSize == 0 or YSize == 0 or nbands == 0:
        
        ndims = out_data.ndim
        if ndims == 2:
            YSize, XSize = np.shape(out_data)
            #print(XSize, YSize, nbands)
            out_data = np.reshape(out_data, (YSize, XSize,-1))  

        YSize, XSize, nbands = np.shape(out_data)
        #print(XSize, YSize, nbands)
        #print(nbands)
    
    if datatype == 1:
        out_ds = driver.Create(output_file,   #filename
                               XSize,     #no. of columns
                               YSize,     # # of rows
                               nbands,              # # of bands
                               gdal.GDT_Byte)  # UInt16, Int16, _Byte
    elif datatype == 2:
        out_ds = driver.Create(output_file,   #filename
                               XSize,     #no. of columns
                               YSize,     # # of rows
                               nbands,              # # of bands
                               gdal.GDT_Int16)  # UInt16, Int16, _Byte
    elif datatype == 3:
        out_ds = driver.Create(output_file,   #filename
                               XSize,     #no. of columns
                               YSize,     # # of rows
                               nbands,              # # of bands
                               gdal.GDT_UInt16)  # UInt16, Int16, _Byte        
    elif datatype == 4:
        out_ds = driver.Create(output_file,   #filename
                               XSize,     #no. of columns
                               YSize,     # # of rows
                               nbands,              # # of bands
                               gdal.GDT_Float32)  # UInt16, Int16, _Byte  
    else:
        out_ds = driver.Create(output_file,   #filename
                               XSize,     #no. of columns
                               YSize,     # # of rows
                               nbands,              # # of bands
                               gdal.GDT_UInt16)  # UInt16, Int16, _Byte

    # top left x, w-e pixel resolution, rotation, top left y, rotation, n-s pixel resolution
    out_ds.SetGeoTransform(geo_transform)##sets same geotransform as input
    out_ds.SetProjection(projection)##sets same projection as input

    # data = np. 
    for i in range(nbands):
     
        band = out_ds.GetRasterBand(i+1)
        if outbands != None:
            band.SetRasterCategoryNames([outbands[i]])
        band.WriteArray(out_data[:,:,i])   #get current band #writting output raster
         #setting nodata value
        #band.SetNoDataValue(-340282346638528859811704183484516925440.000000)
        if novalue != None:
            band.SetNoDataValue(novalue)

    #Close output raster dataset
    out_ds = None


#create a dataset from copy 
def creatFileCopy(dst_filename,src_filename, filetype, out_data, nbands, outbands):

    #driver = gdal.GetDriverByName(filetype)
    driver = gdal.GetDriverByName('GTiff')
    src_ds = gdal.Open( src_filename )
    
    dst_ds = driver.CreateCopy( dst_filename, src_ds, 0 )
    
    for i in range(nbands):
     
        band = dst_ds.GetRasterBand(i+1)
        band.SetRasterCategoryNames([outbands[i]])
        band.WriteArray(out_data[:,:,i])   #get current band #writting output raster
         #setting nodata value
        #band.SetNoDataValue(-340282346638528859811704183484516925440.000000)
        band.SetNoDataValue(-32768)

    #Close output raster dataset
    out_ds = None


'''function to change the datatype of the image'''
def dataTypeTrans(filename, inimaxi = 32767, inimini = 0, filetype = 'GTiff', outdatatype =2,novalue = None, outfile ='out.tif'):

    outbands =[]
    
    print(filename)
    metadata = getMeta(filename) 

    n = metadata['nbands']     #no. of the bands
    width = metadata['width']  #no. of columns of the image
    height = metadata['height'] #no. of rows of the image

    #print(n,width,height)
    GT = metadata['geotransform']  #geographic coordinate 
    projection = metadata['projection']

    basename = filename[:-4]

    bands=range(n)

    raster = getAllData(filename)

    maxival = np.max(raster[raster < inimaxi])
    minival = np.min(raster[raster > 1])
    #print(maxival, minival)

    # the maximum value changes along with the output data type
    # if output data type is 1 (Byte), the maximum value will be 255, novalue will be 0
    # if output data type is 2 (Int16), the maximum value will be 32767, novalue will be -32768
    # if output data type is 3 (UInt16), the maximum value will be 65536, novalue will be 0
    # if output data type is 4 (Float32), which ususlly means transfer rediation to reflectance
    # reflectance range is [0 - 1.0]...............
    if outdatatype == 1:  #datatype
        newmaxival = 255
        outdata = ((raster - minival) /(maxival - minival))*newmaxival
        outdata.astype('byte')
        new_novalue = 0
        
    elif outdatatype == 2:
        newmaxival = 32767
        outdata = ((raster - minival) /(maxival - minival))*newmaxival
        outdata.astype('int16')
        new_novalue = -32768
        
    elif outdatatype == 4:
        newmaxival = 65535
        outdata = ((raster - minival) /(maxival - minival))*newmaxival
        new_novalue = 0
        outdata.astype('uint16')
        
    elif outdatatype == 4:
        newmaxival = 1.0
        outdata = raster /maxival
        new_novalue = 0
        outdata.astype('float32')
        
    else:
        newmaxival = 1.0
        outdata = raster /maxival
        new_novalue = 0
    if novalue == 0:
        novalue = new_novalue
    #outdata = normarr(rasterdata, maxival,minival,255)
    creatFile(output_file = outfile, 
              out_data = outdata, 
              novalue = novalue,
              geo_transform = GT,
              projection =projection)
            


