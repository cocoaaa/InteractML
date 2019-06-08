import math

class VectorTile():
    """
    Assumes tiles are 256x256 pixel
    Also refer to
    https://svn.openstreetmap.org/applications/routing/pyroute/tilenames.py
    """
    
    SIZE = 256
    
    @staticmethod
    def deg2tile_xy(lat_deg, lon_deg, zoom):
        """
        Lat,Lon to tile numbers
        - src: https://is.gd/mjvdR7
        """
        lat_rad = math.radians(lat_deg)
        n = 2.0 ** zoom
        xtile = int((lon_deg + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
        return (xtile, ytile)

    @staticmethod
    def tile_xyz2deg(xtile, ytile, zoom):
        """
        Tile numbers to lat/lon in degree
        This returns the NW-corner of the square. 
        Use the function with xtile+1 and/or ytile+1 to get the other corners. 
        With xtile+0.5 & ytile+0.5 it will return the center of the tile.
        - src: https://is.gd/mjvdR7
        """
        n = 2.0 ** zoom
        lon_deg = xtile / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
        lat_deg = math.degrees(lat_rad)
        return (lat_deg, lon_deg)