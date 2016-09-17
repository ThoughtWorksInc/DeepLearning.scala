libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.0" % Test

classpathTypes += "maven-plugin"

def osClassifier(moduleId: ModuleID) = {
  import org.apache.commons.lang3.SystemUtils
  val arch = SystemUtils.OS_ARCH match {
    case "amd64" => "x86_64"
    case other => other
  }
  if (SystemUtils.IS_OS_MAC_OSX) {
    moduleId classifier s"macosx-${arch}"
  } else if (SystemUtils.IS_OS_LINUX) {
    moduleId classifier s"linux-${arch}"
  } else if (SystemUtils.IS_OS_WINDOWS) {
    moduleId classifier s"windows-${arch}"
  } else {
    moduleId
  }
}

libraryDependencies += osClassifier("org.nd4j" % "nd4j-native" % "0.4-rc3.9" % Test classifier "")