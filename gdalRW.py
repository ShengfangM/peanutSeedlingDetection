#!/usr/bin/env python
# coding: utf-8

# In[1]:


from osgeo import gdal, osr
import numpy as np


# In[2]:


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
#     if geotransform:
#         print("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
#         print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))
# adfGeoTransform[0] /* top left x */
# adfGeoTransform[1] /* w-e pixel resolution */
# adfGeoTransform[2] /* rotation, 0 if image is "north up" */
# adfGeoTransform[3] /* top left y */
# adfGeoTransform[4] /* rotation, 0 if image is "north up" */
# adfGeoTransform[5] /* n-s pixel resolution */

    # Metadata for the raster dataset
    metadata['metadata'] = dataset.GetMetadata()

    band = dataset.GetRasterBand(1)
    metadata['datatype'] = gdal.GetDataTypeName(band.DataType)
    metadata['overviews'] = band.GetOverviewCount()
    metadata['nodata'] = band.GetNoDataValue()
    dataset = None
    return metadata


# In[3]:


#read array from input file
def getData(filename,nbands, xoff, yoff, xsize,ysize):
    
    # Open the file:
    dataset = gdal.Open(filename, gdal.GA_ReadOnly)
  
    # data = np. 
    for i in range(nbands):
        band = dataset.GetRasterBand(i + 1)    #get current band
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


# In[4]:


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


# In[5]:


#save array to an gdal supported file e.g. TIFF, HDF5
def creatFile(output_file, out_data, XSize, YSize, nbands,filetype, datatype,outbands,novalue, geo_transform,projection):
# Create gtif file
    #driver = gdal.GetDriverByName("GTiff")
    driver = gdal.GetDriverByName(filetype)
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
        #band.SetRasterCategoryNames([outbands[i]])
        band.WriteArray(out_data[:,:,i])   #get current band #writting output raster
         #setting nodata value
        #band.SetNoDataValue(-340282346638528859811704183484516925440.000000)
        band.SetNoDataValue(novalue)

    #Close output raster dataset
    out_ds = None


# In[6]:


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


# In[8]:


#     # setting spatial reference of output raster
#     srs = osr.SpatialReference()
#     srs.ImportFromWkt(projection)
#     dst_ds.SetProjection( srs.ExportToWkt() )


# In[ ]:




