libraryDependencies += "com.nativelibs4java" % "javacl" % "1.0.0-RC4"

val lwjglNatives: String = {
  import org.apache.commons.lang3.SystemUtils
  if (SystemUtils.IS_OS_MAC_OSX) {
    "natives-macos"
  } else if (SystemUtils.IS_OS_LINUX) {
    "natives-linux"
  } else if (SystemUtils.IS_OS_WINDOWS) {
    "natives-windows"
  } else {
    throw new MessageOnlyException(
      s"lwjgl does not support ${SystemUtils.OS_NAME}")
  }
}

libraryDependencies += "org.lwjgl" % "lwjgl-opencl" % "3.1.1"

libraryDependencies += "org.lwjgl" % "lwjgl" % "3.1.1"

libraryDependencies += "org.lwjgl" % "lwjgl" % "3.1.1" classifier lwjglNatives % Runtime

// FIXME: remove this line
libraryDependencies += "org.jocl" % "jocl" % "2.0.0"

libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.0" % Test
