import sbt.Keys._
import sbt._

object ProjectsGenerator extends AutoPlugin {

  private val CudaPlatforms = List("linux-x86_64", "macosx-x86_64", "windows-x86_64", "linux-ppc64le")
  private val NativePlatforms = "android-arm" :: "android-x86" :: CudaPlatforms

  override def extraProjects: Seq[Project] =
    generateBackendProjects("native", NativePlatforms) ++
      generateBackendProjects("cuda-8.0", CudaPlatforms) ++
      generateBackendProjects("cuda-7.5", CudaPlatforms)

  private def generateBackendProjects(backendType: String, platforms: List[String]): List[Project] = {
    val nd4jVersion = "0.8.0"

    def generateProject(platform: String, backendType: String): Project = {

      val idItem =
        if (backendType == "cuda-8.0") "cuda-8_0"
        else if (backendType == "cuda-7.5") "cuda-7_5"
        else backendType
      val projectId = s"nd4j-$platform-$idItem"

      Project(projectId, file(".nd4j-platform") / projectId).settings(
        classpathTypes += "maven-plugins",
        libraryDependencies += "org.nd4j" % s"nd4j-$backendType" % nd4jVersion classifier platform excludeAll ExclusionRule(
          organization = "org.bytedeco.javacpp-presets"),
        libraryDependencies += "org.bytedeco.javacpp-presets" % "openblas" % "0.2.19-1.3" classifier platform,
        crossPaths := false,
        name := projectId
      )
    }

    val subProjects = platforms map { platform =>
      generateProject(platform, backendType)
    }

    Project("nd4j-platform", file(".nd4j-platform") / "nd4j-platform")
      .settings(crossPaths := false)
      .dependsOn(subProjects.map(p => p: ClasspathDep[ProjectReference]): _*) :: subProjects
  }
}
