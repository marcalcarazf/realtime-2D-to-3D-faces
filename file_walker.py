import os


def walk( path ):
    """ Use to walk through all objects in a directory.
    Yields either File() or Folder() objects."""
    for f in os.listdir(path):
        if os.path.isfile(path):
            yield File(os.path.join(path, f))
        else:
            yield Folder(os.path.join(path, f))

class PathEntity:
    """ Every object in a directory, file or folders.

    Attributes:
        isFile: True if it's a file (=it's a File() object)
        isDirectory: !isFile (=it's a Folder() object)
        full_path: Full Path to entity
        name: Name of entity
    """
    def __init__(self, path):
        self.isFile = os.path.isfile(path)
        self.isDirectory = not self.isFile
        self.full_path = path
        self.name = os.path.splitext(os.path.basename(self.full_path))[0]

class Folder(PathEntity):
    """
    Extends PathEntity with walk. Use like this:
    for f in folder.walk():
        print(f.name)
        ...
    """
    def walk(self):
        return walk( self.full_path )

class File(PathEntity):
    """ Extends entity file useful file Attributes:

     extension: File extension
     open(mode): Opens the file (use only using "with" keyword!)
    """
    def __init__(self, path):
        super(File, self).__init__(path)
        self.extension = os.path.splitext(self.full_Path)[1]

    def open(self, mode):
        return open(self.full_path, mode)


if __name__ == "__main__":
    star = lambda x: "*" if x else "/"
    
    for obj in walk("/opt/"):
        print(("{}{}:".format(obj.name, star(obj.isFile))))

        if obj.isDirectory:
            for sub_obj in obj.walk():
                print(("\t{}{}".format(sub_obj.name, star(sub_obj.isFile))))
