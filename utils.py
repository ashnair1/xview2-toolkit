# Utils
def wkt2list(objwkts):
    import geodaisy.converters as convert
    poly = [eval(convert.wkt_to_geojson(wkt))['coordinates'] for wkt in objwkts]
    return poly
